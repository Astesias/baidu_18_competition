from utils import ezlog
import sys
        
if __name__ == '__main__':
    log=ezlog('./output/log/log.txt')
    log.flush()
    log.add(' '.join(sys.argv))
    
    try:
      0/0
    except:
      log.err()
    print('Done')