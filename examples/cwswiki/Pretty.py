#-------------------------------------------------
# beautification
#-------------------------------------------------

def nicetime(dt): 
   if (dt < 1): 
     return "%4.0fms"%(1000.0*dt) 
   elif (dt < 60): 
     return "%2.3fs"%(dt) 
   elif (dt < 60*60): 
     return "%2.3fm"%(dt/60) 
   elif (dt < 60*60*24): 
     return "%2.3fh"%(dt/(60*60)) 
   else: 
     return "%2.4fd"%(dt/(60*60*24)) 
 
def nicecount(n): 
  if (n < 10*1000): 
    return "%4u"%(n) 
  elif (n < 10*1000*1000): 
    return "%4uK"%(n/1000) 
  elif (n < 10*1000*1000*1000): 
    return "%4uM"%(n/(1000*1000)) 
  else: 
    return "%4uB"%(n/(1000*1000*1000)) 
