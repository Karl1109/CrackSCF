import logging


def get_logger(process_floder_path, name):
    logger = logging.getLogger(name)

    filename = f'{process_floder_path}/{name}.log'

    fh = logging.FileHandler(filename, mode=''
                                            'w+', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    logger.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


if __name__ == "__main__":
    log1 = get_logger('logger1')
    log2 = get_logger('logger2')

    log1.debug("This is a debug log.")
    log1.info("This is a info log.")
    log1.warning("This is a warning log.")
    log1.error("This is a error log.")
    log1.critical("This is a critical log.")

    log2.debug("This is a debug log.")
    log2.info("This is a info log.")
    log2.warning("This is a warning log.")
    log2.error("This is a error log.")
    log2.critical("This is a critical log.")