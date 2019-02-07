import sys
import logging
import unittest
import numpy as np
import pandas as pd

from smote import ADASYN
from hybridboost import HybridBoost

class TestADASYN(unittest.TestCase):
	def setUp(self):
		self.df = pd.read_csv("pima-indians-diabetes.csv")
		self.adasyn = ADASYN(k=5, imb_threshold=1.0, random_state=0, verbose=False)

		# Create X ans y as numpy-array from imbalanced dataset
		self.X = self.df.filter(regex=("^((?!class).)*$")).as_matrix()
		self.y = self.df.filter(regex=("class")).as_matrix().flatten()

		# # Count the frequency of the outcome
		# unique, counts = np.unique(self.y, return_counts=True)
		# print(np.asarray((unique, counts)).T)

	def test_transform(self):
		pass

	def test_fit_transform(self):
		X_res, y_res = self.adasyn.fit_transform(self.X, self.y)

	def test_oversample(self):
		pass

	def test_generate_samples(self):
		pass

class TestHybridAlgo(unittest.TestCase):
	def setUp(self):
		# Create X ans y as numpy-array from imbalanced dataset
		self.X = [[0., 0., 0.], 
				  [1., 1., 1.], 
				  [2., 2., 2.],
				  [3., 3., 3.], 
				  [4., 4., 4.], 
				  [5., 5., 5.],
				  [6., 6., 6.], 
				  [7., 7., 7.],
				  [8., 8., 8.], 
				  [9., 9., 9.],
				  [10., 10., 10.], 
				  [11., 11., 11.],
				 ]
		self.y = [0,1,0,0,0,0,1,1,1,1,1,1]

		# Create algo instance
		self.hybrid = HybridBoost()

	def test_label_minority(self):
		ret = self.hybrid.label_minority(np.array(self.X), np.array(self.y))
		com = {'Safe': {8, 9, 10, 11}, 'Borderline': {6, 7}, 'Outlier': {1}}
		assert ret == com

	def test_undersample(self):
		labels = self.hybrid.label_minority(np.array(self.X), np.array(self.y))
		ret = self.hybrid.undersample(np.array(self.X), np.array(self.y),labels['Borderline'])
		com = [5]
		assert ret == com


if __name__ == "__main__":
    logging.basicConfig( stream=sys.stderr )
    logging.getLogger( "TestSMOTE" ).setLevel( logging.DEBUG )
    unittest.main()