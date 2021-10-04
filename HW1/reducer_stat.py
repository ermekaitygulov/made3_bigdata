import sys


def update_mean(running_mean, running_count, batch_mean, batch_count):
    batch_mean = (running_mean * running_count + batch_mean * batch_count) / (running_count + batch_count)
    batch_count = running_count + batch_count
    return batch_mean, batch_count


def update_var(running_mean, running_var, running_count, batch_mean, batch_var, batch_count):
    var = (running_var * running_count + batch_var * batch_count) / (running_count + batch_count)
    var += running_count * ((running_mean - batch_mean) / (running_count + batch_count)) ** 2
    return var


if __name__ == '__main__':
    mean = 0
    count = 0
    var = 0
    for i, raw_line in enumerate(sys.stdin):
        line = raw_line.strip()
        loc_mean, loc_var, loc_count = list(map(float, line.split('\t')))

        new_mean, new_count = update_mean(mean, count, loc_mean, loc_count)
        var = update_var(mean, var, count, loc_mean, loc_var, loc_count)
        mean, count = new_mean, new_count
    print(f'map_reduce mean: {mean}')
    print(f'map_reduce var: {var}')
