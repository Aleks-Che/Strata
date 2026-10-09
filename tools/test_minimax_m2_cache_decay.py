"""CPU checks for experimental ordering and speed aggregation."""
import unittest
from tools.check_minimax_m2_cache_decay import metrics, schedule, summarize, PERIODS, NAMES


def result(steps,ms,hits=10,h2d=30):
    return {'decode_forward_tokens':steps,'decode_ms':ms,'request_ms':ms+10,'prefill_ms':10,
            'decode':{'cache_hit_bytes':hits,'h2d_bytes':h2d,'cache_evictions':3,
                      'cache_fill_bytes':4,'pipeline_delivery_ms':ms/2}}


class Experiment(unittest.TestCase):
    def test_latin_order_balances_positions(self):
        runs=schedule(3)
        self.assertEqual(len(runs),9)
        for position in range(3):self.assertEqual(set(p for _,p in runs[position::3]),set(PERIODS))
        for round_no in (1,2,3):self.assertEqual(set(p for r,p in runs if r==round_no),set(PERIODS))

    def test_screening_is_complete_round(self):
        self.assertEqual(schedule(1),[(1,p) for p in PERIODS])

    def test_aggregate_uses_steps_over_time(self):
        m=metrics([result(1,1000),result(9,1000)])
        self.assertEqual(m['decode_tok_s'],5)
        self.assertEqual(m['decode_steps'],10)
        self.assertEqual(m['request_ms'],2020)
        self.assertEqual(m['decode_evictions'],6)

    def test_hit_rate_is_byte_weighted(self):
        m=metrics([result(1,1,hits=100,h2d=100),result(1,1,hits=0,h2d=800)])
        self.assertEqual(m['decode_hit_percent'],10)

    def test_old_binary_excluded_from_speed_medians(self):
        runs=[{'round':0,'period':65536,'results':[result(1,999999)]*4}]
        for r,p in schedule(3):
            runs.append({'round':r,'period':p,'results':[result(10,1000*(r if p==65536 else 1))]*4})
        summary=summarize(runs)
        self.assertEqual([s['name'] for s in summary],['all_requests',*NAMES])
        for item in summary:
            self.assertEqual(item['periods']['65536']['median']['decode_tok_s'],5)
            self.assertEqual(item['periods']['131072']['decode_gain_percent'],100)
            self.assertEqual(len(item['periods']['65536']['samples']),3)


if __name__=='__main__':unittest.main()
