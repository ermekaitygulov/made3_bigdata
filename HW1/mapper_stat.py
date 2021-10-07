#!/usr/bin/python3
import sys


def update_mean(old_mean, old_count, new_value):
    new_mean = (old_count * old_mean + new_value) / (old_count + 1)
    return new_mean, old_count + 1


def update_var(old_mean, old_var, old_count, new_value):
    new_var = old_var * old_count / (old_count + 1)
    new_var += old_count * ((old_mean - new_value) / (old_count + 1)) ** 2
    return new_var


def find_col_idx(header, col):
    col_list = header.split(',')
    col_idx = col_list.index(col)
    return col_idx


def remove_inquote_comma(line):
    quote_sep = line.split('"')
    if len(quote_sep) == 1:
        return line
    new_line = []
    for quote_idx, part in enumerate(quote_sep):
        if quote_idx % 2 != 0:
            # part of line in quotes
            part = part.replace(',', '')
        new_line.append(part)
    return '"'.join(new_line)


def check_inquote_eol(line):
    quote_count = line.count('"')
    if quote_count % 2 == 0:
        return False
    else:
        return True


def catch_unexpected_eol(line, tmp):
    if tmp:
        line = '{}{}'.format(tmp, line)
        tmp = None
    line = remove_inquote_comma(line)
    if check_inquote_eol(line):
        tmp = line
        line = None
    return line, tmp


if __name__ == '__main__':
    PRICE_IDX = 9
    mean = 0
    count = 0
    var = 0
    price_idx = None
    prev_line = None
    for i, raw_line in enumerate(sys.stdin):
        cur_line = raw_line.strip()
        cur_line, prev_line = catch_unexpected_eol(cur_line, prev_line)
        if cur_line is None:
            continue

        price = cur_line.split(',')[PRICE_IDX]
        try:
            price = int(price)
        except ValueError:
            continue
        var = update_var(mean, var, count, price)
        mean, count = update_mean(mean, count, price)
    print('{}\t{}\t{}'.format(mean, var, count))
