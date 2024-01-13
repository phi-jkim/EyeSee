import unittest 
from EyeML import * 

import sys 

class TestUtil(unittest.TestCase): 
    '''
    Testing Strategy for Util
    '''
    def test_img_read(self): 
        # print(sys.path)
        img_greyscale = img_read("test/test_data/greyscale.jpg") 
        # PILLOW has small issues with dimension 
        img_nongreyscale = img_read("test/test_data/nongreyscale.jpg") # non-greyscale 
        assert(img_greyscale.shape[2] == 3) 
        assert(img_nongreyscale.shape[2] == 3) 

    def test_annotation_read(self): 
        annotation = annotation_read("test/test_data/discord_pfp1.txt") 
        assert(len(annotation.shape) == 2) 
        np.testing.assert_array_equal(annotation[0,:], np.array([0, 0.330529, 0.292067, 0.382212, 0.266827]))
        np.testing.assert_array_equal(annotation[1,:], np.array([0, 0.566106, 0.783654, 0.728365, 0.423077]))
        assert(annotation.shape == (2, 5)) 

if __name__ == "__main__": 
    unittest.main() 
