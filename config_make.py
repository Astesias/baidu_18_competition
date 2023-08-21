import os
from pysl import Config, getip

def make_cfg():
    config = {
        'port': 8080,
        'serial_host': "/dev/ttyUSB0",
        'serial_bps': 115200,
        'model_dir': os.path.abspath('./dete_model'),
        'model_json': 'usb.json',
        'input_frame_size': (320,320),
        'server_task': './test/dj/mysite/manage.py',
        'main_task': './Baidu_18_ysl/main.py',
        
        'spray_index': 2,  # [0,1,2]
        'items_kind': 'item2', # [item1,item2,item3]
        
        'item_index': 0  # [0,1,2] abandon
    }

    with Config('./configs.json', base=config) as cfg:
        cfg.clean()

        cfg.add('port')
        cfg.add('serial_bps')
        cfg.add('serial_host')
        cfg.add('model_dir')
        cfg.add('serial_bps')
        cfg.add('input_frame_size')

        cfg.add('spray_index')
        cfg.add('items_kind')
        cfg.add('item_index')        

        cfg.add('host', getip()[0])
        cfg.add('server', f'http://{getip()[0]}:'+str(cfg.port))
        cfg.add('model_json', os.path.join(cfg.model_dir, config['model_json']))

        cfg.add('server_task', os.path.abspath(config.get('server_task')))
        cfg.add('main_task', os.path.abspath(config.get('main_task')))
        
        if os.name!='nt':
          videos=[]
          for i in os.listdir('/dev'):
              if 'video' in i:
                videos.append('/dev/'+i)
          if not videos:
            print("Warning video not found")
          videos.sort(key=lambda x:int(x[-1]))
          cfg.add('videos',videos)
       
          usbflag=False
          for i in os.listdir('/dev'):
              if 'ttyUSB' in i:
                cfg.add('serial_host','/dev/'+i) 
                usbflag=True
                break
          if not usbflag:
            print("Warning usbserial not found")
            config['serial_host']="/dev/ttyPS0"
            print('Warning Using ttyPS0 instead')
                
     

    return cfg

if __name__=='__main__':
    make_cfg()
