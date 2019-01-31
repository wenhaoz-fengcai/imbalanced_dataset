import sys
import logging
import unittest
import numpy as np
import pandas as pd

from smote import ADASYN

class TestADASYN(unittest.TestCase):
	def setUp(self):
		self.df = pd.read_csv("pima-indians-diabetes.csv")
		self.adasyn = ADASYN(k=5, imb_threshold=1.0, random_state=0, verbose=False)

		# Create X ans y as numpy-array from imbalanced dataset
		self.X = self.df.filter(regex=("^((?!class).)*$")).as_matrix()
		self.y = self.df.filter(regex=("class")).as_matrix().flatten()

		# Count the frequency of the outcome
		unique, counts = np.unique(self.y, return_counts=True)
		print(np.asarray((unique, counts)).T)

	def test_transform(self):
		pass

	def test_fit_transform(self):
		X_res, y_res = self.adasyn.fit_transform(self.X, self.y)

	def test_oversample(self):
		pass

	def test_generate_samples(self):
		pass

	def test_ADASYN(self):
		pass


if __name__ == "__main__":
    logging.basicConfig( stream=sys.stderr )
    logging.getLogger( "TestSMOTE" ).setLevel( logging.DEBUG )
    unittest.main()