import unittest
import pandas as pd
from .preprocessing import from_string_to_list

class PreprocessingTest(unittest.TestCase):
    def setUp(self):
        self.dataframe = pd.read_csv("Udacity_AZDIAS_Subset.csv", sep=";")
        self.feat_info = pd.read_csv("AZDIAS_Feature_Summary.csv", sep=";")

    def test_default_widget_size(self):
        print(self.feat_info)

        self.feat_info.assign(missing_or_unknown=from_string_to_list)
        self.assertTrue(True)