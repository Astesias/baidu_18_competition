from pysl import Config,os_enter,easy_request
import os,config_make,time,sys


try:
    config_make.make_cfg()
except:
    Warning('configs make failed')
cfg=Config('./configs.json')

t=time.time()
while 1:
    try:
        config_make.make_cfg()
        cfg=Config('./configs.json')
        assert "USB" in cfg['serial_host'],'Error usbserial not found'
        
        video_num=len(cfg['videos'])
        assert video_num==2 , f'Error videos found {video_num}'
    except:
        if time.time()-t<100:
            print('Retry')
            continue
        else:
            break
    os.chdir('./Baidu_18_ysl')
    sys.path.insert(0,'./')
    from main_pcs import run
    run(Q_Order,cfg,False)