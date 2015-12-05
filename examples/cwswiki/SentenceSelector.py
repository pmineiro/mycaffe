import math
import numpy as np
import pyvw
import sys

import DocGenerator

def crossentlabels (s, l):
  return np.asscalar (np.sum (np.log (1.0 + np.exp (s))) - np.sum (s[l]))

def crossent (sp, sq):
  # (((q Log[p] + (1 - q) Log[1 - p] /. q -> 1/(1 + Exp[-sq]) /. 
  #          p -> 1/(1 + Exp[-sp]) // FullSimplify) /. 
  #       Log[1/(1 + E^x_)] -> -Log[1 + E^x] ) // FullSimplify) /. 
  #   Log[1 + E^-sp] -> Log[1 + E^sp] - sp // FullSimplify

  # (E^sq sp)/(1 + E^sq) - Log[1 + E^sp]

  expsq = np.exp (sq)
  return np.asscalar (
           np.sum (  np.divide (np.multiply (sp, expsq), 1.0 + expsq) 
                   - np.log1p (np.exp (sp))))

def sos2 (v,dv,prefix):
  lower=math.floor(v/dv)
  pos=(v/dv)-lower

  return [ ("%s_%d"%(prefix,lower),1.0-pos), ("%s_%d"%(prefix,lower+1),pos) ]

class SentenceSelector (pyvw.SearchTask):
  def __init__ (self, vw, sch, num_actions, task_data):
    pyvw.SearchTask.__init__ (self, vw, sch, num_actions)

    self.finetuner = task_data['finetuner']
    self.finetunedelay = task_data['finetunedelay']
    self.numtakedowns = 0

    self.test = task_data['test']

    self.KEEP = 1
    self.DISCARD = 2

  def __del__ (self):
    self.vw.finish ()

  def __enter__(self):
    return self

  def __exit__ (self, type, value, traceback):
    self.vw.finish ()

  def _takedown (self, example):
    if not self.test:
      if self.numtakedowns == self.finetunedelay:
        sys.stdout.write("*** starting fine-tuning ***\n")

      self.numtakedowns += 1

      if self.numtakedowns > self.finetunedelay:
        # fine-tune underlying predictor
        # TODO: make sure this is policy, not oracle (?)
        for ii in self.selected:
          self.finetuner.update (example.parts[ii], example.labels)

  def _setup (self, example):
    example.scores = self.finetuner.predict (example.parts)
    example.ensemble = np.sum(example.scores,axis=0)/len(example.scores)
    example.eachevidence = [x - self.finetuner.prior ()
                            for x in example.scores]
    example.ensembleevidence = np.sum(example.eachevidence,axis=0)/len(example.scores)

  def _run (self, example):
    if self.sch.get_search_state() == pyvw.SearchState.INIT_TEST:
      init_run = True
    else:
      init_run = False

    prior = self.finetuner.prior ()
    self.curscore = prior
    self.optimalscore = prior
    self.saveoptimalpos = -1
    self.selected = []

    for pos, score in enumerate (example.scores):
      with self.vw.example ({
          'n' : sos2(math.log(1+pos),1,"bin"),
          'p' : [ ('ce',crossent (score,prior)) ],
          'q' : [ ('ce',crossent (score,example.ensemble)) ],
          #'qflass' : [ ('ce',crossent (example.eachevidence[pos],example.ensembleevidence)) ],
#          'pflass' : sos2(crossent(score[1],prior),4,"ce"),
#          'q' : [ ('ce',crossent (score[1],self.curscore)) ] 
        }) as ex:

        delta=crossentlabels (self.curscore, example.labels) - \
              crossentlabels (score, example.labels)
        if delta > 0:
          best = self.KEEP
        else:
          best = self.DISCARD
        
        pred = self.sch.predict (examples=ex, my_tag=1+pos, oracle=best,
                                 condition=[(1+n,'r') for n in self.selected])

        if pred == self.KEEP:
          self.curscore = score
          self.selected = [ pos ]

        if crossentlabels (self.optimalscore, example.labels) > \
           crossentlabels (score, example.labels):
          self.optimalscore = score
          self.saveoptimalpos = pos 

    myloss = crossentlabels (self.curscore, example.labels) 

    if init_run:
      self.saveloss = myloss
      self.saveoptimalloss = crossentlabels (self.optimalscore, example.labels)
      self.savesimpleloss = crossentlabels (example.scores[0], example.labels)
          
    self.sch.loss (myloss)
    return self.selected
