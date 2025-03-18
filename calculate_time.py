import time

def record_timestamp():
    """
    记录当前的时间戳
    :return: 当前的时间戳（秒）
    """
    return time.time()


def calculate_duration(start_timestamp, end_timestamp):
    """
    根据传入的两个时间戳计算时长
    :param start_timestamp: 开始时间戳（秒）
    :param end_timestamp: 结束时间戳（秒）
    :return: 时长（秒）
    """
    return end_timestamp - start_timestamp
