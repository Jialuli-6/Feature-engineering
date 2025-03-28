from aim5005.features import MinMaxScaler, StandardScaler, LabelEncoder
import numpy as np
import unittest
from unittest.case import TestCase

### TO NOT MODIFY EXISTING TESTS

class TestFeatures(TestCase):
    ### Test for MinMaxScaler
    def test_initialize_min_max_scaler(self):
        scaler = MinMaxScaler()
        assert isinstance(scaler, MinMaxScaler), "scaler is not a MinMaxScaler object"
        
        
    def test_min_max_fit(self):
        scaler = MinMaxScaler()
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        scaler.fit(data)
        assert (scaler.maximum == np.array([1., 18.])).all(), "scaler fit does not return maximum values [1., 18.] "
        assert (scaler.minimum == np.array([-1., 2.])).all(), "scaler fit does not return maximum values [-1., 2.] " 
        
        
    def test_min_max_scaler(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[0., 0.], [0.25, 0.25], [0.5, 0.5], [1., 1.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. All Values should be between 0 and 1. Got: {}".format(result.reshape(1,-1))
        
    def test_min_max_scaler_single_value(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[1.5, 0.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect [[1.5 0. ]]. Got: {}".format(result)
        
    ### Test for StandardScaler    
    def test_standard_scaler_init(self):
        scaler = StandardScaler()
        assert isinstance(scaler, StandardScaler), "scaler is not a StandardScaler object"
        
    def test_standard_scaler_get_mean(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([0.5, 0.5])
        scaler.fit(data)
        assert (scaler.mean == expected).all(), "scaler fit does not return expected mean {}. Got {}".format(expected, scaler.mean)

    def test_standard_scaler_get_std(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([0.5, 0.5])
        scaler.fit(data)
        assert (scaler.std == expected).all(), "scaler fit does not return expected std {}. Got {}".format(expected, scaler.std)
        
    def test_standard_scaler_transform(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[-1., -1.], [-1., -1.], [1., 1.], [1., 1.]])
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))
        
    def test_standard_scaler_single_value(self):
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[3., 3.]])
        scaler = StandardScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))

    # TODO: Add a test of your own below this line

    ### Test for LabelEncoder
    def test_initialize_labelencoder(self):
        encoder = LabelEncoder()
        assert isinstance(encoder, LabelEncoder), "encoder is not a LabelEncoder object"

    def test_fit(self):
        """
        Test if LabelEncoder learns the correct unique classes. Rank by alphabetical order.
        """
        encoder = LabelEncoder()
        data = ["Paris", "Tokyo", "Paris", "Shanghai", "Shanghai", "Amsterdam"]
        expected = np.array(["Amsterdam", "Paris", "Shanghai", "Tokyo"])
        encoder.fit(data)
        result = encoder.classes_
        assert (result == expected).all(), "Label encoder does not return expected classes. Expect {}. Got: {}".format(expected, result)

    def test_transform(self):
        """
        Test if LabelEncoder correctly transforms labels into numerical values.
        """
        encoder = LabelEncoder()
        data = ["Paris", "Tokyo", "Paris", "Shanghai", "Shanghai", "Amsterdam"]
        expected = np.array([1, 3, 1, 2, 2, 0])
        encoder.fit(data)
        result = encoder.transform(data)
        assert (result == expected).all(), "Label encoder does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))

    def test_fit_transform(self):
        """
        Test if fit_transform combines fit and transform.
        """
        encoder = LabelEncoder()
        data = ["Paris", "Tokyo", "Paris", "Shanghai", "Shanghai", "Amsterdam"]
        expected = np.array([1, 3, 1, 2, 2, 0]) 
        result = encoder.fit_transform(data)
        assert (result == expected).all(), "Label encoder does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))


    def test_encoder_single_value(self):
        """
        Test with a single example.
        """
        encoder = LabelEncoder()
        data = ["Paris", "Tokyo", "Paris", "Shanghai", "Shanghai", "Amsterdam"]
        expected = np.array([2])
        encoder.fit(data)
        result = encoder.transform(["Shanghai"])
        assert (result == expected).all(), "Scaler transform does not return the expected value. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))

    
if __name__ == '__main__':
    unittest.main()


### run "pip3 install --upgrade --force-reinstall pytest"
### run "python3 -m pytest -s --color=yes" for test