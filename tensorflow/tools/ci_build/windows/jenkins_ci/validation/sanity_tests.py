"""
This file contains the sanity tests need to be run after we build a
TensorFlow wheel.

Check the BKM to get more information
https://wiki.ith.intel.com/display/intelnervana/BKM%3A+Public+CI+Monitoring+on+Windows#BKM:PublicCIMonitoringonWindows-HowtoVerifyifapackageisinstalledandruntestsonit
"""

import tensorflow as tf

print("Sanity test start.")

print("Validate keras")
assert "_v2.keras" in tf.keras.__name__, "Test _v2.keras in tf.keras.__name__"

print("Validate array shape after adding")
t1 = tf.constant([1, 2, 3, 4])
t2 = tf.constant([5, 6, 7, 8])
assert tf.add(t1, t2).shape == (4,), "Test array shape after adding two arrays"

print("Validate tf.estimator")
assert "_v2.estimator" in tf.estimator.__name__, "Test _v2.estimator in tf.estimator.__name__"

print("Sanity test completed.")