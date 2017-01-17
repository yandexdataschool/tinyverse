
Universe RL trainer platform. Simple. Supple. Scalable.

## Why should i care?

tinyverse is a reinforcement learning platform for gym/universe/custom environments that lets you utilize any resources you have to train reinforcement learning algorithm.

### Key features
* __Simple:__ the core is currently under 400 lines including code (~50%), comments(~40%) and spaces (~10%).
* __Supple:__ tinyverse assumes almost nothing of your agent and environment. The environment may not be interruptable. Agent may have any algorithm/structure. Agent _[will soon]_(https://github.com/yandexdataschool/tinyverse/issues/14) support any framework from numpy to pure tensorflow/theano to keras/lasagne+agentnet.
* __Scalable:__ You can train and play 10 parallel games on your GPU desktop/server, 20 more sessions on your Macbook and another 5 on your friend's laptop when he doesn't look. (And 1000 more games and 10 trainers in the cloud ofc).

The core idea is to have two types of processes:
* __play__-er - interacts with the environment, records sessions to the database, periodically loads new params
* __train__-er - reads sessions from the database, trains agent via experience replay, sends params to the database

Those processes revolve around __database__ that stores experience sessions and weights. The database is currently implemented with [Redis](http://redis.io/) since it is simple to setup and swift with key-value operations. You can, however, implement the database [interface](https://github.com/yandexdataschool/tinyverse/blob/master/tinyverse/database.py#L76) with what database you prefer.

![img](https://s23.postimg.org/cei1cd2iz/tinyverse_scheme.png)


### Quickstart

* install redis server
  * (Ubuntu) sudo apt-get install redis-server 
  * Mac OS version [HERE](http://jasdeep.ca/2012/05/installing-redis-on-mac-os-x/). 
  * Otherwise search "Install redis your_OS" or ask us
  * If you want to run on multiple machines, configure redis-server to listen to 0.0.0.0 (also mb set password)
  
* install python packages
  * assuming you have [gym](https://github.com/openai/gym#installing-everything),[universe](https://github.com/openai/universe),[theano,lasagne,agentnet](http://agentnet.readthedocs.io/en/master/user/install.html) (all bleeding edge as of 2016.01.17)
  
```
#create several player processes. Each process plays games and saves results
#the loop below spawns 10 players. If you are doing this on a laptop, reducing to 2-4 is okay
for i in `seq 1 10`; 
do
        THEANO_FLAGS=device=cpu,floatX=float32 python tinyverse atari.py play -b 3 &
done

#create learner process on GPU. batch size 10
THEANO_FLAGS=device=gpu,floatX=float32 python tinyverse atari.py train -b 10 &

#play 5 games, show results, record video to ./records (run once in a while)
python tinyverse atari.py eval -n 5

#dev: see workbench.ipynb
```
