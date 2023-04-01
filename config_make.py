import os
from pysl import Config, getip

config = {
    'port': 1881,
    'serial_host': "/dev/ttyPS0",
    'serial_bps': 115200,
    'model_dir': os.path.abspath('./model'),
    'input_frame_size': (128, 128),
}


with Config('./configs.json', base=config) as cfg:
    cfg.clean()

    cfg.add('port')
    cfg.add('serial_host')
    cfg.add('serial_bps')
    cfg.add('model_dir')
    cfg.add('serial_bps')
    cfg.add('input_frame_size')

    cfg.add('host', getip()[0])
    cfg.add('server', f'http://{getip()[0]}:'+str(cfg.port))
