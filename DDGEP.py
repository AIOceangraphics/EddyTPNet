import numpy as np
from netCDF4 import Dataset
import math

'''
Based on the Cartesian coordinate system, due east is the positive direction
'''


def get_Degree(lon_1, lat_1, lon_2, lat_2):
    y = lat_2 - lat_1
    x = lon_2 - lon_1
    if (y >= 0) & (x > 0):
        return math.degrees(math.atan(y / x))
    if (y < 0) & (x < 0):
        return (180 + math.degrees(math.atan(y / x)))
    if (y < 0) & (x > 0):
        return (360 + math.degrees(math.atan(y / x)))
    if (y >= 0) & (x < 0):
        return (180 + math.degrees(math.atan(y / x)))
    if (x == 0) & (y > 0):
        return 90
    if (x == 0) & (y < 0):
        return 270
    if (x == 0) & (y == 0):
        return 180


"""
draw Angular Divergence Grid Map
"""


def Grid_Map(data):
    file = 'AE_Angular_Divergence_Grid_Map.grd'
    with open(file, 'a') as f:
        f.write('DSAA\n')
        f.write('2880 1440\n')
        f.write('0 360\n')
        f.write('-90 90\n')
        f.write('0 360\n')  ##数据范围
        for m in range(1440):
            for n in range(2880):
                f.write(str(data[m, n]) + ' ')
            f.write('\n')


'''
Fill grid data
'''


def fill_Grid(list_lon, list_lat, data_grid, data_index):
    list_degree = []
    list_degree_lon = []
    list_degree_lat = []
    '''
    clear data
    '''
    for i in range(len(list_lon)):
        if list_lon[i] < 0:
            list_lon[i] = 360 + list_lon[i]
        if list_lon[i] >= 360:
            list_lon[i] = list_lon[i] - 360
        if list_lat[i] == 90:
            del list_lon[i]
            del list_lat[i]
        else:
            pass
        list_lat[i] = list_lat[i] + 90

    if len(list_lon) >= 30:
        for i in range(2, len(list_lon) - 2, 1):
            degree = get_Degree(
                (list_lon[i - 2] + list_lon[i - 1] + list_lon[i]) / 3,
                (list_lat[i - 2] + list_lat[i - 1] + list_lat[i]) / 3,
                (list_lon[i + 1] + list_lon[i + 2] + list_lon[i]) / 3,
                (list_lat[i + 1] + list_lat[i + 2] + list_lat[i]) / 3,
            )
            list_degree.append(degree)
            list_degree_lon.append(list_lon[i])
            list_degree_lat.append(list_lat[i])

        for i in range(len(list_degree)):
            lon_p = int((list_degree_lon[i] - int(list_degree_lon[i])) / 0.125)
            lat_p = int((list_degree_lat[i] - int(list_degree_lat[i])) / 0.125)
            lat = int(list_degree_lat[i]) * 8 + lat_p
            lon = int(list_degree_lon[i]) * 8 + lon_p
            data_grid[lat, lon] = data_grid[lat, lon] + list_degree[i]
            data_index[lat, lon] = data_index[lat, lon] + 1

    return data_grid, data_index


def save_Grid(path, save_path):
    directory_name = path
    nc_obj = Dataset(directory_name)

    data_grid = np.zeros((1440, 2880))
    data_index = np.zeros((1440, 2880))

    m = 1
    list_lon = []
    list_lat = []

    for i in nc_obj['track'][1:]:
        if nc_obj['track'][m] == nc_obj['track'][m - 1]:
            list_lon.append(nc_obj['longitude'][m])
            list_lat.append(nc_obj['latitude'][m])
        else:
            data_grid, data_index = fill_Grid(list_lon, list_lat, data_grid, data_index)
            print(m)
            list_lon = []
            list_lat = []
            list_lon.append(nc_obj['longitude'][m])
            list_lat.append(nc_obj['latitude'][m])
            m = m + 1
    data_grid, data_index = fill_Grid(list_lon, list_lat, data_grid, data_index)

    # average
    for i in range(1440):
        for j in range(2880):
            if data_index[i][j] > 0:
                data_grid[i][j] = data_grid[i][j] / data_grid[i][j]
            else:
                pass

    np.save(save_path, data_grid)
