from torch import optim
from Data_loader import *
from GD_loss import *
from GD import *
from Moudle import *
from Stand_Normal import *
import datetime

def train_test(data_path,DDGEP_path):
    """
    angle,latitudinal displacement,meridional displacement,season,amplitude,radius,,speed_average,latitude,longitude
    """
    row_data = np.load(data_path)
    num = 120442
    all_angle = np.load(DDGEP_path)

    angle = row_data[:, :, 0:1]
    for i in range(len(angle)):
        for j in range(27):
            angle[i, j, 0] = int(angle[i, j, 0])
    lat_lon_dis = row_data[:, :, 1:3]
    season = row_data[:, :, 3:4]
    energy = row_data[:, :, 4:7]
    lat_lon = row_data[:, :, 7:9]

    # normalization standardization
    lat_lon_stand, lat_lon_mean, lat_lon_std = Standardize(lat_lon)
    lat_lon_dis_stand, lat_lon_dis_mean, lat_lon_dis_std = Standardize(lat_lon_dis)
    energy_standar, energy_mean, energy_std = Standardize(energy)
    angle_normal, angle_max, angle_min = Normalization(angle)

    # season,latitude,longitude,latitudinal displacement,meridional displacement,amplitude,radius,speed_average,angle
    Data = np.concatenate((season, lat_lon_stand, lat_lon_dis_stand, energy_standar, angle_normal), axis=2)
    train_data = Data[0:84310, 0:20, :]
    train_label = Data[0:84310, 20:27, :]
    test_data = Data[84310:, 0:20, :]
    test_label = Data[84310:, 20:27, :]

    batch_size = 512

    energy_encoder_input_size = 3
    energy_seq_len = 20
    energy_hidden_size = 64
    energy_num_layers = 8

    encoder_input_size = 8
    encoder_seq_len = 20
    encoder_hidden_size = 128
    encoder_num_layers = 8

    decoder_input_size = 4 + 4 + 128 + 64 + 1
    decoder_hidden_size = 128
    output_size = 2
    decoder_num_layers = 8

    prediction_days = 7

    epochs = 50
    batch_first = True
    lr = 0.00001

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loader_train, loader_test = Data_loader(train_data, train_label, test_data, test_label, batch_size)

    energy_encoder_net = Energy_EncoderNet(energy_seq_len, energy_encoder_input_size, energy_hidden_size,
                                           energy_num_layers,
                                           batch_first)
    encoder_net = Encoder_Net(encoder_seq_len, encoder_input_size, encoder_hidden_size, encoder_num_layers, batch_first)
    decoder_net = Decoder_Net(decoder_input_size, decoder_hidden_size, output_size, decoder_num_layers, batch_first)

    energy_encoder_net = energy_encoder_net.to(device)
    encoder_net = encoder_net.to(device)
    decoder_net = decoder_net.to(device)

    criterion1 = nn.modules.L1Loss(reduction='mean').to(device)
    criterion2 = GDLoss().to(device)

    optimizer = optim.Adam([{'params': energy_encoder_net.parameters(), 'lr': lr},
                            {'params': encoder_net.parameters(), 'lr': lr},
                            {'params': decoder_net.parameters(), 'lr': lr}, ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)  # 修改学习率

    """
    record loss
    """
    loss_train = np.zeros((epochs))

    startTime = datetime.datetime.now()
    """
    train
    """
    for epoch in range(epochs):

        energy_encoder_net.train()
        encoder_net.train()
        decoder_net.train()

        epoch_train_loss = 0

        for i, (encoder_data, encoder_label) in enumerate(loader_train):

            energy_encoder_h0 = torch.zeros(energy_num_layers, encoder_data.shape[0], energy_hidden_size)
            encoder_h0 = torch.zeros(encoder_num_layers, encoder_data.shape[0], encoder_hidden_size)
            encoder_c0 = torch.zeros(encoder_num_layers, encoder_data.shape[0], encoder_hidden_size)

            energy_batch_data = encoder_data[:, :, 5:8]
            lon_lat_encoder_batch_data = encoder_data[:, :, 1:5]
            time_encoder_batch_data = encoder_data[:, :, 0:1]
            angle_encoder_batch_data = encoder_data[:, :, 8:9]
            """
            latitude,longitude,latitudinal displacement,meridional displacement,season,angle
            """
            encoder_batch_data = torch.cat((lon_lat_encoder_batch_data,
                                            time_encoder_batch_data,
                                            angle_encoder_batch_data), dim=2).to(device)

            """
            label
            """
            decoder_batch_label_lon_lat = encoder_label[:, :, 1:5]
            decoder_batch_label_time = encoder_label[:, :, 0:1]
            decoder_batch_label_angle = encoder_label[:, :, 8:9]
            decoder_batch_label = torch.cat((decoder_batch_label_lon_lat,
                                             decoder_batch_label_time,
                                             decoder_batch_label_angle), dim=2).to(device)
            """
            encode
            """
            energy_out = energy_encoder_net(
                energy_batch_data.to(device),
                energy_encoder_h0.to(device), )
            energy_encoder_out = torch.zeros((len(encoder_batch_data), 5, energy_hidden_size))

            for o in range(5):
                energy_encoder_out[:, o:o + 1, :] = energy_out

            encoder_out, encoder_hn, encoder_cn = encoder_net(
                lon_lat_encoder_batch_data.to(device),
                time_encoder_batch_data.to(device),
                encoder_h0.to(device),
                encoder_c0.to(device))

            encoder_out_ = torch.zeros((len(encoder_batch_data), 5, encoder_hidden_size))

            for o in range(5):
                encoder_out_[:, o:o + 1, :] = encoder_out

            _loss = torch.zeros(1).to(device)

            decoder_hn = encoder_hn.to(device)
            decoder_cn = encoder_cn.to(device)

            previous_data = torch.cat((encoder_batch_data[:, 15:20, 0:6].to(device),
                                       decoder_batch_label[:, :, 0:6].to(device)),
                                      dim=1).to(device)

            site = previous_data[:, 4:11, 0:2]

            for j in range(prediction_days):
                temp_previous_data = previous_data[:, j:j + 5, ...].clone()
                temp_site = site[:, j, :]
                """
                decode
                """
                decoder_output, hn, cn = decoder_net(
                    torch.cat((energy_encoder_out.to(device),
                               encoder_out_.to(device),
                               temp_previous_data[:, :, 0:4].to(device)), dim=2).to(device),
                    temp_previous_data[:, :, 4:5].to(device),
                    temp_previous_data[:, :, 5:6].to(device),
                    decoder_hn.to(device),
                    decoder_cn.to(device))

                temp_label = decoder_batch_label[:, j:j + 1, 2:4]

                temp_loss1 = criterion1(decoder_output, temp_label)

                decoder_output = decoder_output.reshape((len(decoder_output), 2))
                temp_label = temp_label.reshape((len(temp_label), 2))

                temp_loss2 = criterion2(decoder_output, temp_label, lat_lon_dis_mean, lat_lon_dis_std,
                                        temp_site, lat_lon_mean, lat_lon_std, device)

                _loss = temp_loss1 + temp_loss2 + _loss

                decoder_hn = hn
                decoder_cn = cn

            _loss = _loss / prediction_days

            epoch_train_loss = epoch_train_loss + _loss.item()
            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()

        scheduler.step(epoch_train_loss / len(loader_train))
        epoch_train_loss = epoch_train_loss / len(loader_train)

        print('train_loss_{}:'.format(epoch), epoch_train_loss)

        loss_train[epoch] = epoch_train_loss

    endTime = datetime.datetime.now()
    print("run time：%ss" % (endTime - startTime).seconds)

    # save model
    torch.save(encoder_net, 'encoder_net.pth')
    torch.save(decoder_net, 'decoder_net.pth')
    torch.save(energy_encoder_net, 'energy_encoder_net.pth')
    # save loss
    np.save('loss_train.npy', loss_train)

    print('*' * 100)

    """
    test
    """

    encoder_net = torch.load('encoder_net.pth')
    decoder_net = torch.load('decoder_net.pth')
    energy_encoder_net = torch.load('energy_encoder_net.pth')

    def count_param(model):
        param_count = 0
        for param in model.parameters():
            param_count += param.view(-1).size()[0]
        return param_count

    print(count_param(decoder_net))
    print(count_param(encoder_net))
    print(count_param(energy_encoder_net))

    energy_encoder_net.eval()
    encoder_net.eval()
    decoder_net.eval()

    count = 0
    predict_data = torch.zeros((len(test_label), 7, 2))

    epoch_test_loss = 0

    startTime = datetime.datetime.now()
    for i, (encoder_data, encoder_label) in enumerate(loader_test):
        with torch.no_grad():

            energy_encoder_h0 = torch.zeros(energy_num_layers, encoder_data.shape[0], energy_hidden_size)
            encoder_h0 = torch.zeros(encoder_num_layers, encoder_data.shape[0], encoder_hidden_size)
            encoder_c0 = torch.zeros(encoder_num_layers, encoder_data.shape[0], encoder_hidden_size)

            energy_batch_data = encoder_data[:, :, 5:8]
            lon_lat_encoder_batch_data = encoder_data[:, :, 1:5]
            time_encoder_batch_data = encoder_data[:, :, 0:1]
            angle_encoder_batch_data = encoder_data[:, :, 8:9]

            encoder_batch_data = torch.cat((lon_lat_encoder_batch_data,
                                            time_encoder_batch_data,
                                            angle_encoder_batch_data), dim=2).to(device)

            # label
            decoder_batch_label_lon_lat = encoder_label[:, :, 1:5]
            decoder_batch_label_time = encoder_label[:, :, 0:1]
            decoder_batch_label_angle = encoder_label[:, :, 8:9]
            decoder_batch_label = torch.cat((decoder_batch_label_lon_lat,
                                             decoder_batch_label_time,
                                             decoder_batch_label_angle), dim=2).to(device)

            # encode
            energy_out = energy_encoder_net(
                energy_batch_data.to(device),
                energy_encoder_h0.to(device), )
            energy_encoder_out = torch.zeros((len(encoder_batch_data), 5, energy_hidden_size))
            for o in range(5):
                energy_encoder_out[:, o:o + 1, :] = energy_out

            encoder_out, encoder_hn, encoder_cn = encoder_net(
                lon_lat_encoder_batch_data.to(device),
                time_encoder_batch_data.to(device),
                encoder_h0.to(device),
                encoder_c0.to(device))
            encoder_out_ = torch.zeros((len(encoder_batch_data), 5, encoder_hidden_size))
            for o in range(5):
                encoder_out_[:, o:o + 1, :] = encoder_out

            _loss = torch.zeros(1).to(device)

            decoder_hn = encoder_hn.to(device)
            decoder_cn = encoder_cn.to(device)

            predict = torch.zeros((len(encoder_batch_data), 7, 6))

            predict[:, :, 4:5] = decoder_batch_label_time

            # decoder
            for j in range(prediction_days):
                previous_data = torch.cat((encoder_batch_data[:, 15:20, 0:6].to(device), predict[:, :, :].to(device)),
                                          dim=1).to(
                    device)
                temp_previous_data = previous_data[:, j:j + 5, ...].clone()

                site = previous_data[:, 4:11, 0:2]

                temp_site = site[:, j, :]

                decoder_output, hn, cn = decoder_net(
                    torch.cat((energy_encoder_out.to(device),
                               encoder_out_.to(device),
                               temp_previous_data[:, :, 0:4].to(device)), dim=2).to(device),
                    temp_previous_data[:, :, 4:5].to(device),
                    temp_previous_data[:, :, 5:6].to(device),
                    decoder_hn.to(device),
                    decoder_cn.to(device))

                predict[:, j:j + 1, 2:4] = decoder_output

                temp_decoder_output = decoder_output.clone()
                temp_decoder_output = temp_decoder_output.reshape((len(temp_decoder_output), 2))
                temp_decoder_output = abStandardize(temp_decoder_output.cpu(), lat_lon_dis_mean, lat_lon_dis_std)
                new_temp_site = temp_site.clone()
                new_temp_site = abStandardize(new_temp_site.cpu(), lat_lon_mean, lat_lon_std)
                new_site = new_temp_site + temp_decoder_output

                normal_new_site = (new_site - lat_lon_mean) / lat_lon_std
                normal_new_site = torch.tensor(normal_new_site, dtype=torch.float64)
                normal_new_site = normal_new_site.reshape((len(normal_new_site), 1, 2))
                predict[:, j:j + 1, 0:2] = normal_new_site[:, :, :]

                new_site[:, 0:1] = new_site[:, 0:1] + 90

                temp_angle = torch.zeros((len(new_site), 1))
                for q in range(len(new_site)):
                    lat_p = int((new_site[q, 0] - int(new_site[q, 0])) / 0.125)
                    lon_p = int((new_site[q, 1] - int(new_site[q, 1])) / 0.125)
                    lat = int(new_site[q, 0]) * 8 + lat_p
                    lon = int(new_site[q, 1]) * 8 + lon_p
                    temp_angle[q, 0] = all_angle[lat, lon]

                temp_angle_normal = (temp_angle - angle_min) / (angle_max - angle_min)

                temp_angle_normal = torch.tensor(temp_angle_normal, dtype=torch.float64)
                temp_angle_normal = temp_angle_normal.reshape((len(temp_angle_normal), 1, 1))
                predict[:, j:j + 1, 5:6] = temp_angle_normal[:, :, :]

                temp_label = decoder_batch_label[:, j:j + 1, 2:4]

                temp_loss1 = criterion1(decoder_output, temp_label)
                decoder_output = decoder_output.reshape((len(decoder_output), 2))
                temp_label = temp_label.reshape((len(temp_label), 2))
                new_site = new_site.reshape((len(new_site), 2))

                temp_loss2 = criterion2(decoder_output.to(device), temp_label.to(device), lat_lon_dis_mean,
                                        lat_lon_dis_std,
                                        temp_site.to(device), lat_lon_mean, lat_lon_std, device)

                _loss = temp_loss1 + temp_loss2 + _loss

                decoder_hn = hn
                decoder_cn = cn

            _loss = _loss / prediction_days

            epoch_test_loss = epoch_test_loss + _loss.item()

        predict_data[count:count + len(predict)] = predict[:, :, 0:2]

        count = count + len(predict)

    epoch_test_loss = epoch_test_loss / len(loader_test)

    endTime = datetime.datetime.now()
    print("run time：%ss" % (endTime - startTime).seconds)

    predict_data[:, :, 0:2] = abStandardize(predict_data[:, :, 0:2], lat_lon_mean, lat_lon_std)
    predict_data = predict_data.detach().numpy()
    test_label = row_data[84310:, 20:27, 7:9]

    epoch_GD_test_loss = GD(test_label, predict_data)

    print('test_loss_{}:', epoch_test_loss)
    print('epoch_GD_test_loss_{}:', epoch_GD_test_loss)

    # save
    np.save('predict_data.npy', predict_data[:, :, 0:2])
    np.save('test_data.npy', row_data[84310:, 0:20, 7:9])
    np.save('test_label.npy', row_data[84310:, 20:27, 7:9])
