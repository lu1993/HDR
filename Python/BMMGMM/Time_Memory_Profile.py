# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 21:03:39 2016

@author: lcao
"""
# time and memory profile 
%load_ext memory_profiler
%load_ext line_profiler
from memory_profiler import profile

# record total memory and time used
import time
start = time.time()
%memit -r 1 model.fit model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)
end = time.time()
print(end - start)

# record memory usage and time line by line
#%mprun -f model.fit model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)
#%lprun -f model.fit model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
