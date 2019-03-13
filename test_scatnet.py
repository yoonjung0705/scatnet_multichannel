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

	# FIXME: add tests on optimize_filter() and filter_freq()

	def test_map_meta(self):
		'''
		- test if copying columns matches what is expected for the following cases:
		1 column to 1 column
		3 columns to 3 columns
		2 columns to 4 columns
		
		when from_meta and to_meta have an overlap of key value pairs while also having
		nonoverlapping key value pairs
		
		when to_meta has no key value pairs
		when to_meta has no overlapping key value pairs
		when from_meta has no key value pairs
		2 columns to 1 column (should raise error)
		when index is out of bound for from_meta (should raise error)
		when index is out of bound for to_meta
		when from_meta and to_meta have an overlap of key value pairs while
		a) to_meta having empty list of indices
		b) from_meta having empty list of indices
		when "exclude" argument is not empty
		from_meta is intact after function call

		FIXME: for shared keys, if to_ind goes out of bound, should to_meta's shared key be
		extended to incorporate that? or should it raise an error? Current version does not extend
		'''
		# 1 column to 1 column
		from_ind = 2
		to_ind = 3
		from_meta = self.create_meta(('key1', (8,5)), ('key2', (10,5)))
		to_meta = self.create_meta(('key1', (4,5)), ('key3', (4,5)))
		to_meta_orig = to_meta.copy()
		to_meta_out = sn.map_meta(from_meta, from_ind, to_meta, to_ind, exclude=None)
		# check if to_meta's shared key's to_ind values are identical to 
		# from_meta's shared key's from_ind values 
		self.assertTrue(np.isclose(from_meta['key1'][from_ind], 
			to_meta_out['key1'][to_ind], rtol=1e-5, atol=1e-8).all())
		# check if to_meta's shared key's indices NOT IN to_ind are identical to 
		# to_meta_orig's shared key's indices NOT IN to_ind 
		to_ind_intact = [idx for idx in range(len(to_meta_orig['key1'])) if idx != to_ind]
		self.assertTrue(np.isclose(to_meta_orig['key1'][to_ind_intact], 
			to_meta_out['key1'][to_ind_intact], rtol=1e-5, atol=1e-8).all())
		# check if to_meta's pure key's values are identical to 
		# to_meta_orig's pure key's values
		self.assertTrue(np.isclose(to_meta_orig['key3'], 
			to_meta_out['key3'], rtol=1e-5, atol=1e-8).all())
		# check if from_meta's pure key's values are copied into to_meta
		self.assertTrue(np.isclose(from_meta['key2'][from_ind], 
			to_meta_out['key2'][to_ind], rtol=1e-5, atol=1e-8).all())

		# 3 columns to 3 columns
		from_ind = [2,1,0]
		to_ind = [1,3,2]
		from_meta = self.create_meta(('key1', (7,5)), ('key2', (12,5)))
		to_meta = self.create_meta(('key1', (6,5)), ('key3', (8,5)))
		to_meta_orig = to_meta.copy()
		to_meta_out = sn.map_meta(from_meta, from_ind, to_meta, to_ind, exclude=None)
		# check if to_meta's shared key's to_ind values are identical to 
		# from_meta's shared key's from_ind values 
		self.assertTrue(np.isclose(from_meta['key1'][from_ind], 
			to_meta_out['key1'][to_ind], rtol=1e-5, atol=1e-8).all())
		# check if to_meta's shared key's indices NOT IN to_ind are identical to 
		# to_meta_orig's shared key's indices NOT IN to_ind 
		to_ind_intact = [idx for idx in range(len(to_meta_orig['key1'])) if idx not in to_ind]
		self.assertTrue(np.isclose(to_meta_orig['key1'][to_ind_intact], 
			to_meta_out['key1'][to_ind_intact], rtol=1e-5, atol=1e-8).all())
		# check if to_meta's pure key's values are identical to 
		# to_meta_orig's pure key's values
		self.assertTrue(np.isclose(to_meta_orig['key3'], 
			to_meta_out['key3'], rtol=1e-5, atol=1e-8).all())
		# check if from_meta's pure key's values are copied into to_meta
		self.assertTrue(np.isclose(from_meta['key2'][from_ind], 
			to_meta_out['key2'][to_ind], rtol=1e-5, atol=1e-8).all())

		# 2 columns to 4 columns
		from_ind = [2,1]
		to_ind = [1,3,2,0]
		from_meta = self.create_meta(('key1', (7,5)), ('key2', (12,5)))
		to_meta = self.create_meta(('key1', (6,5)), ('key3', (8,5)))
		to_meta_orig = to_meta.copy()
		to_meta_out = sn.map_meta(from_meta, from_ind, to_meta, to_ind, exclude=None)
		# check if to_meta's shared key's to_ind values are identical to 
		# from_meta's shared key's from_ind values 
		self.assertTrue(np.isclose(np.tile(from_meta['key1'][from_ind], 
			(int(len(to_ind)/len(from_ind)), 1)), 
			to_meta_out['key1'][to_ind], rtol=1e-5, atol=1e-8).all())
		# check if to_meta's shared key's indices NOT IN to_ind are identical to 
		# to_meta_orig's shared key's indices NOT IN to_ind 
		to_ind_intact = [idx for idx in range(len(to_meta_orig['key1'])) if idx not in to_ind]
		self.assertTrue(np.isclose(to_meta_orig['key1'][to_ind_intact], 
			to_meta_out['key1'][to_ind_intact], rtol=1e-5, atol=1e-8).all())
		# check if to_meta's pure key's values are identical to 
		# to_meta_orig's pure key's values
		self.assertTrue(np.isclose(to_meta_orig['key3'], 
			to_meta_out['key3'], rtol=1e-5, atol=1e-8).all())
		# check if from_meta's pure key's values are copied into to_meta
		self.assertTrue(np.isclose(np.tile(from_meta['key2'][from_ind], 
			(int(len(to_ind)/len(from_ind)), 1)),
			to_meta_out['key2'][to_ind], rtol=1e-5, atol=1e-8).all())


		# when to_meta has no key value pairs, try 3 cols to 3 cols
		from_ind = [2,1,0]
		to_ind = [1,3,2]
		from_meta = self.create_meta(('key1', (7,5)), ('key2', (12,5)))
		to_meta = {}
		to_meta_orig = to_meta.copy()
		to_meta_out = sn.map_meta(from_meta, from_ind, to_meta, to_ind, exclude=None)
		# check if from_meta's pure key's values are copied into to_meta
		self.assertTrue(np.isclose(from_meta['key1'][from_ind], 
			to_meta_out['key1'][to_ind], rtol=1e-5, atol=1e-8).all())
		self.assertTrue(np.isclose(from_meta['key2'][from_ind], 
			to_meta_out['key2'][to_ind], rtol=1e-5, atol=1e-8).all())

		# when there are no shared keys, 3 columns to 3 columns
		from_ind = [2,1,0]
		to_ind = [1,3,2]
		from_meta = self.create_meta(('key1', (7,5)), ('key2', (12,5)))
		to_meta = self.create_meta(('key3', (6,5)), ('key4', (8,5)))
		to_meta_orig = to_meta.copy()
		to_meta_out = sn.map_meta(from_meta, from_ind, to_meta, to_ind, exclude=None)
		# check if to_meta's pure key's values are identical to 
		# to_meta_orig's pure key's values
		self.assertTrue(np.isclose(to_meta_orig['key3'], 
			to_meta_out['key3'], rtol=1e-5, atol=1e-8).all())
		self.assertTrue(np.isclose(to_meta_orig['key4'], 
			to_meta_out['key4'], rtol=1e-5, atol=1e-8).all())
		# check if from_meta's pure key's values are copied into to_meta
		self.assertTrue(np.isclose(from_meta['key1'][from_ind], 
			to_meta_out['key1'][to_ind], rtol=1e-5, atol=1e-8).all())
		self.assertTrue(np.isclose(from_meta['key2'][from_ind], 
			to_meta_out['key2'][to_ind], rtol=1e-5, atol=1e-8).all())

		# when from_meta has no key value pairs, try 1 col to 1 col
		from_ind = 2
		to_ind = 3
		from_meta = {}
		to_meta = self.create_meta(('key1', (4,5)), ('key2', (4,5)))
		to_meta_orig = to_meta.copy()
		to_meta_out = sn.map_meta(from_meta, from_ind, to_meta, to_ind, exclude=None)
		# check if to_meta's pure key's values are identical to 
		# to_meta_orig's pure key's values
		self.assertTrue(np.isclose(to_meta_orig['key1'], 
			to_meta_out['key1'], rtol=1e-5, atol=1e-8).all())
		self.assertTrue(np.isclose(to_meta_orig['key2'], 
			to_meta_out['key2'], rtol=1e-5, atol=1e-8).all())

		# 2 columns to 1 column (should raise error)
		from_ind = [2,1]
		to_ind = 1
		from_meta = self.create_meta(('key1', (7,5)), ('key2', (12,5)))
		to_meta = self.create_meta(('key1', (3,5)), ('key3', (6,5)))
		to_meta_orig = to_meta.copy()
		# following should raise error
		with self.assertRaises(ValueError):
			to_meta_out = sn.map_meta(from_meta, from_ind, to_meta, to_ind, exclude=None)

		# when index is out of bound for from_meta (should raise error)
		from_ind = [20,1]
		to_ind = [1,2]
		from_meta = self.create_meta(('key1', (7,5)), ('key2', (12,5)))
		to_meta = self.create_meta(('key1', (3,5)), ('key3', (6,5)))
		to_meta_orig = to_meta.copy()
		# following should raise error
		with self.assertRaises(IndexError):
			to_meta_out = sn.map_meta(from_meta, from_ind, to_meta, to_ind, exclude=None)

		# when index is out of bound for to_meta
		from_ind = [3,1]
		to_ind = [20,2]
		from_meta = self.create_meta(('key1', (7,5)), ('key2', (12,5)))
		to_meta = self.create_meta(('key1', (3,5)), ('key3', (6,5)))
		to_meta_orig = to_meta.copy()
		to_meta_out = sn.map_meta(from_meta, from_ind, to_meta, to_ind, exclude=None)
		# check if to_meta's shared key's to_ind values are identical to 
		# from_meta's shared key's from_ind values 
		self.assertTrue(np.isclose(from_meta['key1'][from_ind], 
			to_meta_out['key1'][to_ind], rtol=1e-5, atol=1e-8).all())
		# check if to_meta's shared key's indices NOT IN to_ind are identical to 
		# to_meta_orig's shared key's indices NOT IN to_ind 
		to_ind_intact = [idx for idx in range(len(to_meta_orig['key1'])) if idx not in to_ind]
		self.assertTrue(np.isclose(to_meta_orig['key1'][to_ind_intact], 
			to_meta_out['key1'][to_ind_intact], rtol=1e-5, atol=1e-8).all())
		# check if to_meta's pure key's values are identical to 
		# to_meta_orig's pure key's values
		self.assertTrue(np.isclose(to_meta_orig['key3'], 
			to_meta_out['key3'], rtol=1e-5, atol=1e-8).all())
		# check if from_meta's pure key's values are copied into to_meta
		self.assertTrue(np.isclose(from_meta['key2'][from_ind], 
			to_meta_out['key2'][to_ind], rtol=1e-5, atol=1e-8).all())

		# when from_meta having empty list of indices, check if there's no change in to_meta
		from_ind = []
		to_ind = [1,2]
		from_meta = self.create_meta(('key1', (7,5)), ('key2', (12,5)))
		to_meta = self.create_meta(('key1', (3,5)), ('key3', (6,5)))
		to_meta_orig = to_meta.copy()
		to_meta_out = sn.map_meta(from_meta, from_ind, to_meta, to_ind, exclude=None)		
		# following should return to_meta without any change
		for key in to_meta.keys():
			self.assertTrue(np.isclose(to_meta_orig[key], to_meta_orig[key], 
				rtol=1e-5, atol=1e-8).all())
			
		# when to_meta having empty list of indices, check if there's no change in to_meta
		from_ind = [1,2]
		to_ind = []
		from_meta = self.create_meta(('key1', (7,5)), ('key2', (12,5)))
		to_meta = self.create_meta(('key1', (3,5)), ('key3', (6,5)))
		to_meta_orig = to_meta.copy()
		to_meta_out = sn.map_meta(from_meta, from_ind, to_meta, to_ind, exclude=None)		
		# following should return to_meta without any change
		for key in to_meta.keys():
			self.assertTrue(np.isclose(to_meta_orig[key], to_meta_orig[key], 
				rtol=1e-5, atol=1e-8).all())

		# when "exclude" argument is not empty
		from_ind = [2,1,0]
		to_ind = [1,3,2]
		from_meta = self.create_meta(('key1', (7,5)), ('key2', (12,5)), ('key3', (4,5)))
		to_meta = self.create_meta(('key1', (6,5)), ('key2', (8,5)), ('key4', (16,5)))
		to_meta_orig = to_meta.copy()
		exclude = ['key2']
		to_meta_out = sn.map_meta(from_meta, from_ind, to_meta, to_ind, exclude=exclude)
		# check if to_meta's shared key's to_ind values are identical to 
		# from_meta's shared key's from_ind values 
		self.assertTrue(np.isclose(from_meta['key1'][from_ind], 
			to_meta_out['key1'][to_ind], rtol=1e-5, atol=1e-8).all())
		# check if to_meta's shared key's indices NOT IN to_ind are identical to 
		# to_meta_orig's shared key's indices NOT IN to_ind 
		to_ind_intact = [idx for idx in range(len(to_meta_orig['key1'])) if idx not in to_ind]
		self.assertTrue(np.isclose(to_meta_orig['key1'][to_ind_intact], 
			to_meta_out['key1'][to_ind_intact], rtol=1e-5, atol=1e-8).all())
		# check if to_meta's pure key's values are identical to 
		# to_meta_orig's pure key's values
		self.assertTrue(np.isclose(to_meta_orig['key4'], 
			to_meta_out['key4'], rtol=1e-5, atol=1e-8).all())
		# check if from_meta's pure key's values are copied into to_meta
		self.assertTrue(np.isclose(from_meta['key3'][from_ind], 
			to_meta_out['key3'][to_ind], rtol=1e-5, atol=1e-8).all())
		# check if key in exclude list is not affected in to_meta
		self.assertTrue(np.isclose(to_meta_orig['key2'], 
			to_meta_out['key2'], rtol=1e-5, atol=1e-8).all())

		# from_meta is intact after function call
		from_ind = [2,1,0]
		to_ind = [1,3,2]
		from_meta = self.create_meta(('key1', (7,5)), ('key2', (12,5)))
		to_meta = self.create_meta(('key1', (6,5)), ('key3', (8,5)))
		from_meta_orig = from_meta.copy()
		to_meta_out = sn.map_meta(from_meta, from_ind, to_meta, to_ind, exclude=None)
		for key in from_meta.keys():
			self.assertTrue(np.isclose(from_meta_orig[key], 
				from_meta[key], rtol=1e-5, atol=1e-8).all())

		# when shape[1] is different (should raise error)
		# 1 column to 1 column
		from_ind = 2
		to_ind = 3
		from_meta = self.create_meta(('key1', (8,7)), ('key2', (10,3)))
		to_meta = self.create_meta(('key1', (4,5)), ('key3', (4,5)))
		to_meta_orig = to_meta.copy()

		with self.assertRaises(ValueError):
			to_meta_out = sn.map_meta(from_meta, from_ind, to_meta, to_ind, exclude=None)

	def create_meta(self, *args):
		'''
		helper function for generating dict type objects used for testing map_meta()
		inputs:
		-------
		- args: length 2 tuple describing a single key-value pair to be added to the dict
		example: create_meta(('key1', (7,5)), ('key2', (12,5)), ('key3', (3,5)))
		will create a dict type object with keys being key1, key2, key3, and the corresponding
		values being np arrays with shape (7,5), (12,5), (3,5), respectively
		'''
		meta = {}
		for kv in args:
			key, shape = kv
			meta[key] = np.random.random(shape)
		return meta	


if __name__ == '__main__':
	unittest.main()