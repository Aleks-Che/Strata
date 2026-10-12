import unittest
from tools.analyze_deepseek4_routes import analyze


class RoutesTest(unittest.TestCase):
    def row(self, ids, **changes):
        return dict(owner='target', matrix='layer3', hash=False, tokens=len(ids),
                    prefill=False, ids=ids, **changes)

    def test_no_future_training_and_batch_union(self):
        rows = [self.row([[1, 2]]), self.row([[3, 4], [1, 5]])]
        r = analyze(rows)['methods']['previous']
        self.assertEqual(r['offered'], 2)
        self.assertEqual(r['first_hits'], 0)
        self.assertEqual(r['useful'], 1)
        self.assertEqual(r['union_targets'], 4)

    def test_layer_model_isolation_and_hash_exclusion(self):
        rows = [self.row([[1]]), self.row([[2]])]
        rows[1]['owner'] = 'draft'
        row = self.row([[1]]); row['hash'] = True; rows.append(row)
        result = analyze(rows)
        self.assertEqual(result['learned_batches'], 0)
        self.assertEqual(result['hash_rows_skipped'], 1)

    def test_prefill_trains_but_is_not_scored(self):
        row = self.row([[8, 9]]); row['prefill'] = True
        r = analyze([row, self.row([[8, 9]])])
        self.assertEqual(r['methods']['previous']['precision'], 1)
        self.assertFalse(r['live_prefetch_enabled'])
        self.assertEqual(r['screening_candidates'], {})


if __name__ == '__main__':
    unittest.main()
