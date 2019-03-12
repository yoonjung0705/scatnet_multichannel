import unittest
import scatnet as sn

class ScatnetTestCase(unittest.TestCase):
	def test_T_to_J(self):
		'''
		- test if J is list when audio and scalar when dyadic
		'''
		# FIXME: check if Q,J,B are altogether list or scalar at the same time
		# this test is carried out assuming this is true		
		for T in [10, 100, 1000, 10000]:
			s = sn.default_filter_opt('audio', 5)
			J = sn.T_to_J(T, s)
			self.assertTrue(isinstance(J, list))

			s = sn.default_filter_opt('dyadic', 5)
			J = sn.T_to_J(T, s)
			self.assertTrue(isinstance(J, (int, float)))

	def test_default_filter_opt(self):
		'''
		- test if Q and J are present as keys
		- test if Q and J are either length 2 list or number
		'''
		# self.assertRaises(sn.default_filter_opt('image', 10))
		for filter_type in ['audio', 'dyadic']:
			s = sn.default_filter_opt(filter_type, 5)
			self.assertTrue('Q' in s.keys())
			self.assertTrue('J' in s.keys())
			if isinstance(s['Q'], list):
				self.assertTrue(len(s['Q'])==2)
			else:
				self.assertTrue(s['Q'] > 0)

			if isinstance(s['J'], list):
				self.assertTrue(len(s['J'])==2) # FIXME: check if J field being list means length being 2
			else:
				self.assertTrue(s['J'] > 0)

	def test_fill_struct(self):
		'''
		- test if key value pair added when key not present
		- test if key value pair not added when key present
		'''
		s = {'key1':1, 'key2':3}
		sn.fill_struct(s, key1=1)
		sn.fill_struct(s, key2=2)
		sn.fill_struct(s, key3=3)
		self.assertEqual(s['key1'], 1)
		self.assertEqual(s['key2'], 3)
		self.assertEqual(s['key3'], 3)

if __name__ == '__main__':
	unittest.main()