import unittest
import numpy as np
import scatnet as sn

class ScatnetTestCase(unittest.TestCase):
	def test_T_to_J(self):
		'''
		- test if J is list when audio and scalar when dyadic
		- test if argument filt_opt remains instact upon function call
		FIXME: check if Q,J,B are altogether list or scalar at the same time
		'''
		for T in [10, 100, 1000, 10000]:
			s = sn.default_filter_opt('audio', 5)
			J = sn.T_to_J(T, s)
			self.assertIsInstance(J, list)

			s = sn.default_filter_opt('dyadic', 5)
			J = sn.T_to_J(T, s)
			self.assertIsInstance(J, (int, float))

		s = sn.default_filter_opt('audio', 10)
		s_cp = s.copy()
		J = sn.T_to_J(T, s)
		self.assertEqual(s, s_cp)

	def test_default_filter_opt(self):
		'''
		- test if Q and J are present as keys
		- test if Q and J are either length 2 list or number
		'''
		# self.assertRaises(sn.default_filter_opt('image', 10))
		for filter_type in ['audio', 'dyadic']:
			s = sn.default_filter_opt(filter_type, 5)
			self.assertIn('Q', s.keys())
			self.assertIn('J', s.keys())
			if isinstance(s['Q'], list):
				self.assertEqual(len(s['Q']), 2)
			else:
				self.assertTrue(s['Q'] > 0)

			if isinstance(s['J'], list):
				self.assertEqual(len(s['J']), 2) # FIXME: check if J field being list means length being 2
			else:
				self.assertTrue(s['J'] > 0)

	# FIXME: add tests on periodize_filter(), optimize_filter()
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

	def test_morlet_freq_1d(self):
		'''
		- test if xi_psi, bw_psi are both type list.
		- test if bw_phi and all elements in bw_psi elements are positive. 
		NOTE: xi_psi can have negative elements. The stepsize in xi_psi in the linearly spaced
		spectrum can in theory be negative if sigma_phi is small although for parameters that
		construct phi this does not happen. However, even though the stepsize is positive for normal
		inputs, the number of steps taken linearly towards the negative frequency regime can result
		in negative values of center frequencies
		
		REVIEW: confirm whether the parameters that result in negative center frequencies are feasible
		If not, no need to test for cases having negative center frequencies  

		- test if xi_psi, bw_psi have length J+P, J+P+1, respectively
		- test if filt_opt does not change upon function call
		'''
		filt_opt = {'xi_psi':0.5, 'sigma_psi':0.4, 'sigma_phi':0.5, 'J':11,
			'Q':8, 'P':5, 'phi_dirac': True}
		# retain a copy of filt_opt to confirm no change upon function call
		filt_opt_cp = filt_opt.copy() 
		xi_psi, bw_psi, bw_phi = sn.morlet_freq_1d(filt_opt)
		self.assertIsInstance(xi_psi, list)
		self.assertIsInstance(bw_psi, list)
		# self.assertTrue(all([xi > 0 for xi in xi_psi]))
		self.assertTrue(all([bw > 0 for bw in bw_psi]))
		self.assertTrue(bw_phi > 0)
		self.assertEqual(len(xi_psi), filt_opt['J'] + filt_opt['P'])
		self.assertEqual(len(bw_psi), filt_opt['J'] + filt_opt['P'] + 1)
		self.assertEqual(filt_opt, filt_opt_cp)





if __name__ == '__main__':
	unittest.main()