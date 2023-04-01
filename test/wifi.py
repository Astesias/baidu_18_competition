from flask import Flask,render_template
import time
from multiprocessing import Process as pcs
from multiprocessing import Queue as queue

class WifiSender():
    def __init__(self,maxsize=5):
        self.T=T=time.time()
        self.Q=Q=queue.Queue(maxsize=maxsize)

        self.app=app=Flask(__name__)
        app.jinja_env.auto_reload = True
        app.config['TEMPLATES_AUTO_RELOAD'] = True
        app.config['SECRET_KEY'] = 'iamthekey'

        @app.route('/')
        def root():
            return render_template('log.html')

        @app.route('/data<string:t>')
        def data(t):
            if Q.qsize():
                t=f'{t[:-3]}.{t[-2:]}'

                last=float(t)-T 
                return f'[{last:.0f}s]'+Q.get()
            else:
                return 'NoData'
            
        # Thread(target=self.run).start()

    def put(self,msg):
            self.Q.put(msg)

    def run(self,host='0.0.0.0',post='1881'):
         self.app.run(host,post)
    
    def get_Q(self):
         return self.Q



def pcd_tasker(Q):
    from flask import Flask,render_template
    from threading import Thread
    import time

    class WifiSender():
        def __init__(self,Q):
            self.T=T=time.time()
            self.Q=Q

            self.app=app=Flask(__name__)
            app.jinja_env.auto_reload = True
            app.config['TEMPLATES_AUTO_RELOAD'] = True
            app.config['SECRET_KEY'] = 'iamthekey'

            @app.route('/')
            def root():
                return render_template('log.html')

            @app.route('/data<string:t>')
            def data(t):
                if Q.qsize():
                    t=f'{t[:-3]}.{t[-2:]}'
                    last=float(t)-T 
                    return f'[{last:.2f}s]{Q.get()}'
                else:
                    return 'NoData'
                

        def put(self,msg):
                self.Q.put(msg)

        def run(self,host='0.0.0.0',post='1881'):
            self.app.run(host,post)
        
        
    sender=WifiSender(Q)
    Thread(target=sender.run).start()
    while 1:
         sender.put(Q.get())



if __name__=='__main__':
    #  sender=WifiSender()
    #  Thread(target=sender.run).start()
    #  while 1:
    #     sender.put('555555555555555')
    #     time.sleep(1)

    Q=queue(maxsize=5)
    pcs(target=pcd_tasker,args=[Q]).start()
    while 1:
         Q.put(time.time())
         time.sleep(1)
     






































