import os
import sys

from unittest import TestCase

# Add root of the dir to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../')))

from trainer import Trainer

class TrainerTest(TestCase):
    
    def setUp(self):
        self.logs_dir = './logs'
        self.trainer = Trainer(None, None, None, None, logs_dir=self.logs_dir)

    def tearDown(self):
        os.remove(self.trainer._get_log_file_path('train'))
        os.rmdir(self.logs_dir)

    def test_logs_dir(self):
        # It created the directory
        self.assertTrue(os.path.exists(self.logs_dir))
        # Write some random logs
        for i in range(50):
            self.trainer.log(i//10, i, 100, 0.9**i, 'train')
        # Get the logs
        logs = self.trainer.load_logs('train')
        self.assertEqual(logs.shape[0], 50)
        self.assertEqual(logs.shape[1], 5)
        # Check first line
        self.assertEqual(logs[0, 1].item(), 0)
        self.assertEqual(logs[0, 2].item(), 0)
        self.assertEqual(logs[0, 3].item(), 100)
        self.assertEqual(logs[0, 4].item(), 1)