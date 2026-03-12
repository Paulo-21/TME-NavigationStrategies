#!/usr/bin/env python

from radarGuidance import *
from wallFollower import *

import random #used for the random choice of a strategy
import sys
import numpy as np
import math
import time
import pickle
#--------------------------------------
# Position of the goal:
goalx = 300
goaly = 450
# Initial position of the robot:
initx = 300
inity = 35
# strategy choice related stuff:
choice = -1
choice_tm1 = -1
tLastChoice = 0
lasttime = time.time()
rew = 0
stepptime = int(2.0/0.01)
lasttimestep = 0
i2name=['wallFollower','radarGuidance']

# Parameters of State      σ ( z ) j = e z j ∑ k = 1 K e z k {\displaystyle \sigma (\mathbf {z} )_{j}={\frac {\mathrm {e} ^{z_{j}}}{\sum _{k=1}^{K}\mathrm {e} ^{z_{k}}}}} {\displaystyle \sigma (\mathbf {z} )_{j}={\frac {\mathrm {e} ^{z_{j}}}{\sum _{k=1}^{K}\mathrm {e} ^{z_{k}}}}} pour tout j ∈ { 1 , … , K } {\displaystyle j\in \left\{1,\ldots ,K\right\}} {\displaystyle j\in \left\{1,\ldots ,K\right\}},building:
# threshold for wall consideration
th_neglectedWall = 35
# threshold to consider that we are too close to a wall
# and a punishment should be delivered
th_obstacleTooClose = 13
# angular limits used to define states
angleLMin = 0
angleLMax = 55

angleFMin=56
angleFMax=143

angleRMin=144
angleRMax=199

# Q-learning related stuff:
# definition of states at time t and t-1
S_t = ''
S_tm1 = ''
Qtable = {}
statetocoord = {}
epsilon = 0.90
alpha = 0.4
beta = 4
gamma = 0.95

def getAfromQ(state):
    global Qtable, beta
    q_vals = Qtable[state]
    max_q = max(q_vals)
    exp_q = [math.exp(beta * (q - max_q)) for q in q_vals]
    total = sum(exp_q)
    probs = [x / total for x in exp_q]
    return np.random.choice([0, 1], p=probs)

def argmax_action(state):
	global Qtable
	if Qtable[state][0] > Qtable[state][1]:
		return Qtable[state][0]
	else:
		return Qtable[state][1]

def updateQtable(state, next_state, reward, action):
	global Qtable
	global gamma
	global alpha
	#print("update , r ",reward)
	#print("Last  ",Qtable[state][action], " -> ",Qtable[state][action]+ alpha * (reward +  gamma * argmax_action(next_state) - Qtable[state][action]))
	Qtable[state][action] += alpha * (reward +  gamma * argmax_action(next_state) - Qtable[state][action])

#--------------------------------------
# the function that selects which controller (radarGuidance or wallFollower) to use
# sets the global variable "choice" to 0 (wallFollower) or 1 (radarGuidance)
# * arbitrationMethod: how to select? 'random','randPersist','qlearning'
def strategyGating(arbitrationMethod,trialnow, verbose=True, state=None):
  global choice
  global choice_tm1
  global tLastChoice
  global rew
  global lasttime
  global lasttrial
  global lasttimestep
  global epsilon

  # The chosen gating strategy is to be coded here:
  #------------------------------------------------
  if arbitrationMethod=='random':
    choice = random.randrange(2)
  #------------------------------------------------
  elif arbitrationMethod=='randomPersist':
  	#print(time.time() - lasttime)
  	#if ( time.time() - lasttime ) >= 2:
  	if trialnow - lasttimestep >= stepptime:
  		lasttimestep = trialnow
  		choice = random.randrange(2)

  #------------------------------------------------
  elif arbitrationMethod=='qlearning':
    #print('Q-Learning selection : to be implemented')
    choice = getAfromQ(state)
     #print("choix : ", choice)

  #------------------------------------------------
  else:
    print(arbitrationMethod+' unknown.')
    exit()

  if verbose:
    print("strategyGating: Active Module: "+i2name[choice])

#--------------------------------------
def buildStateFromSensors(laserRanges,radar,dist2goal):
  S   = ''
  # determine if obstacle on the left:
  wall='0'
  if min(laserRanges[angleLMin:angleLMax]) < th_neglectedWall:
    wall ='1'
  S += wall
  # determine if obstacle in front:
  wall='0'
  if min(laserRanges[angleFMin:angleFMax]) < th_neglectedWall:
    wall ='1'
    #print("Mur Devant")
  S += wall
  # determine if obstacle on the right:
  wall='0'
  if min(laserRanges[angleRMin:angleRMax]) < th_neglectedWall:
    wall ='1'
  S += wall

  S += str(radar)

  if dist2goal < 125:
    S+='0'
  elif dist2goal < 250:
    S+='1'
  else:
    S+='2'
  #print('buildStateFromSensors: State: '+S)

  return S

#--------------------------------------
def main():
  global S_t
  global S_tm1
  global rew
  global lasttimestep
  global choice
  settings = Settings('worlds/entonnoir.xml')
  lastchoice = 0
  timelastchoice = 0
  env_map = settings.map()
  robot = settings.robot()

  d = Display(env_map, robot)
  #method = 'random'
  #method = 'randomPersist'
  method = 'qlearning'
  # experiment related stuff
  startT = time.time()
  trial = 0
  nbTrials = 40
  trialDuration = np.zeros((nbTrials))
  step = 0
  i = 0
  while trial<nbTrials:
    # update the display
    #-------------------------------------
     #if trial > nbTrials - 20:
    #d.update()
    # get position data from the simulation
    #-------------------------------------
    pos = robot.get_pos()
    # print("##########\nStep "+str(i)+" robot pos: x = "+str(int(pos.x()))+" y = "+str(int(pos.y()))+" theta = "+str(int(pos.theta()/math.pi*180.)))

    # has the robot found the reward ?
    #------------------------------------
    dist2goal = math.sqrt((pos.x()-goalx)**2+(pos.y()-goaly)**2)
    # if so, teleport it to initial position, store trial duration, set reward to 1:


    if (dist2goal<20): # 30
      print('***** REWARD REACHED *****')
      pos.set_x(initx)
      pos.set_y(inity)
      robot.set_pos(pos) # format ?
      # and store information about the duration of the finishing trial:
      currT = time.time()
      trialDuration[trial] = currT - startT
      startT = currT
      print("Trial "+str(trial)+" duration:"+str(trialDuration[trial]))
      trial +=1
      step = 0
      rew = 1
      lasttimestep=0
    elif step > 1700 and method == "qlearning":
        print('***** RESET REACHED *****')
        pos.set_x(initx)
        pos.set_y(inity)
        robot.set_pos(pos) # format ?
        # and store information about the duration of the finishing trial:
        currT = time.time()
        trialDuration[trial] = currT - startT
        startT = currT
        print("Trial "+str(trial)+" duration:"+str(trialDuration[trial]))
        #trial +=1
        step = 0
        rew = 0
        lasttimestep=0
    # get the sensor inputs:
    #------------------------------------
    lasers = robot.get_laser_scanners()[0].get_lasers()
    laserRanges = []
    for l in lasers:
      laserRanges.append(l.get_dist())

    radar = robot.get_radars()[0].get_activated_slice()
    bumperL = robot.get_left_bumper()
    bumperR = robot.get_right_bumper()
    # 2) has the robot bumped into a wall ?
    #------------------------------------
    if bumperR or bumperL or min(laserRanges[angleFMin:angleFMax]) < th_obstacleTooClose:
      rew = -1
     # print("***** BING! ***** "+i2name[choice])

    # 3) build the state, that will be used by learning, from the sensory data
    #------------------------------------
    S_tm1 = S_t
    S_t = buildStateFromSensors(laserRanges,radar, dist2goal)
    if S_tm1 not in Qtable:
      Qtable[S_tm1] = [0,0]
      #print("NOT IN")
    if S_t not in Qtable:
      Qtable[S_t] = [0,0]
      #print("NOT IN")

    #print(S_tm1, " ", S_t," ",  Qtable[S_tm1])
    if S_tm1 != S_t:
        if S_t not in statetocoord:
            statetocoord[S_t] = [(int(pos.x()), int(pos.y()))]
        elif (int(pos.x()), int(pos.y())) not in statetocoord[S_t] :
            statetocoord[S_t].append((int(pos.x()), int(pos.y())))

    if S_tm1 != '' and (S_tm1 != S_t or rew != 0 or (step - lasttimestep) >= stepptime) and method == "qlearning":
      updateQtable(S_tm1,S_t, rew, choice)
      rew = 0
      lasttimestep = step
      strategyGating(method,step, verbose=False, state = S_tm1)
      if choice != lastchoice:
        lastchoice = choice
        timelastchoice = step
      elif choice == lastchoice and step-timelastchoice > stepptime:
        if choice == 1:
          choice = 0
        else:
          choice = 1
        timelastchoice = step
      lastchoice = choice
    #------------------------------------
    if method != "qlearning":
        strategyGating(method,step, verbose=False, state = S_tm1)
    if choice==0:
      v = wallFollower(laserRanges,verbose=False)
      #print("WALLFollower")
    else:
      #print("radarGuidance")
      v = radarGuidance(laserRanges,bumperL,bumperR,radar,verbose=False)
    step+=1

    i+=1
    robot.move(v[0], v[1], env_map)
    """if trial > nbTrials - 20:
        time.sleep(0.01)"""
  # When the experiment is over:
  np.savetxt('log/'+str(startT)+'-TrialDurations-'+method+'.txt',trialDuration)
  #np.savetxt('log/'+str(startT)+'-TrialDurations-'+method+'.txt',trialDuration.mean)
  if method == "qlearning":
    with open('log/'+str(startT)+'Qtable.txt','wb') as data:
        pickle.dump(Qtable, data)
    with open('log/'+str(startT)+'Qtable_str.txt','w') as data:
        data.write(str(Qtable))

    with open('log/'+str(startT)+'StateToPos.txt','wb') as data:
        pickle.dump(statetocoord, data)
        #data.write(str(statetocoord))
#--------------------------------------

if __name__ == '__main__':
    for i in range(10):
        random.seed()
        main()
