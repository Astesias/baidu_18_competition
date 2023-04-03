import os
from pysl import Config, getip

def make_cfg():
    config = {
        'port': 1881,
        'serial_host': "/dev/ttyPS0",
        'serial_bps': 115200,
        'model_dir': os.path.abspath('./model'),
        'model_json': 'usb_yolov3.json',
        'input_frame_size': (128, 128),
        'server_task': './test/dj/mysite/sender.py',
        'main_task': './Baidu_18_ysl/main.py',
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
        cfg.add('model_json', os.path.join(cfg.model_dir, config['model_json']))

        cfg.add('server_task', os.path.abspath(config.get('server_task')))
        cfg.add('main_task', os.path.abspath(config.get('main_task')))

    return cfg

if __name__=='__main__':
    make_cfg()