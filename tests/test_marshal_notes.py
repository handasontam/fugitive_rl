import unittest
import numpy as np
from envs.fugitive import MarshalNotes, MAX_HAND_SIZE, multibinary_to_cards

class TestMarshalNotes(unittest.TestCase):

    def setUp(self):
        self.notes = MarshalNotes()

    def test_initial_state(self):
        self.assertEqual(self.notes.n_hideouts, 1)
        self.assertEqual(len(self.notes.graph.nodes()), 1)
        self.assertTrue((0, 0) in self.notes.graph.nodes())
        self.assertTrue(np.array_equal(self.notes.hideouts_range[0], np.array([1] + [0]*42)))

    def test_add_new_hideout(self):
        self.notes.add_new_hideout(n_sprint=1, filtered_vals=[])
        self.assertEqual(self.notes.n_hideouts, 2)
        self.assertTrue(any(node[0] == 1 for node in self.notes.graph.nodes()))
        self.assertTrue(np.any(self.notes.hideouts_range[1]))

    def test_eliminate_vals(self):
        self.notes.add_new_hideout(n_sprint=1, filtered_vals=[])
        self.notes.eliminate_vals([3, 4])
        self.assertFalse(any(node[1] == 3 for node in self.notes.graph.nodes() if node[0] > 0))
        self.assertFalse(any(node[1] == 4 for node in self.notes.graph.nodes() if node[0] > 0))
        self.assertEqual(self.notes.hideouts_range[:, 3].sum(), 0)
        self.assertEqual(self.notes.hideouts_range[:, 4].sum(), 0)

    def test_hideout_revealed(self):
        self.notes.add_new_hideout(n_sprint=1, filtered_vals=[])
        self.notes.hideout_revealed(depth=1, hideout=5)
        self.assertTrue((1, 5) in self.notes.graph.nodes())
        self.assertEqual(self.notes.hideouts_range[1].sum(), 1)
        self.assertEqual(self.notes.hideouts_range[1, 5], 1)

    def test_hideout_revealed_multiple_hideouts(self):
        self.notes.add_new_hideout(n_sprint=1, filtered_vals=[])
        self.notes.add_new_hideout(n_sprint=2, filtered_vals=[])
        self.notes.hideout_revealed(depth=1, hideout=4)
        self.assertTrue((1, 4) in self.notes.graph.nodes())
        self.assertEqual(self.notes.hideouts_range[1].sum(), 1)
        self.assertEqual(self.notes.hideouts_range[1, 4], 1)
        expected_range = [[0], [4], [5, 6, 7, 8, 9, 10, 11]]
        for i in range(3):
            self.assertTrue(np.array_equal(multibinary_to_cards(self.notes.hideouts_range[i]), expected_range[i]))

    def test_get_hideouts_range(self):
        hideouts_range = self.notes.get_hideouts_range()
        self.assertEqual(hideouts_range.shape, (MAX_HAND_SIZE, 43))
        self.assertTrue(np.array_equal(hideouts_range[0], np.array([1] + [0]*42)))

    def test_multiple_hideouts(self):
        self.notes.add_new_hideout(n_sprint=1, filtered_vals=[])
        self.notes.add_new_hideout(n_sprint=2, filtered_vals=[])
        self.assertEqual(self.notes.n_hideouts, 3)
        self.assertTrue(any(node[0] == 2 for node in self.notes.graph.nodes()))

    def test_unreachable_nodes_removal(self):
        self.notes.add_new_hideout(n_sprint=1, filtered_vals=[])
        self.notes.add_new_hideout(n_sprint=1, filtered_vals=[])
        self.notes.eliminate_vals([2, 3])
        self.notes.eliminate_vals([4])
        expected_range = [[0], [1, 5], [5, 6, 7, 8, 9, 10]]
        for i in range(3):
            self.assertTrue(np.array_equal(multibinary_to_cards(self.notes.hideouts_range[i]), expected_range[i]))

if __name__ == '__main__':
    unittest.main()