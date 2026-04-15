from collections import defaultdict
import numpy as np

class ResultMerger:
    def __init__(self):
        self.all_sum = defaultdict(float)
        self.all_cnt = 0

        self.level1_sum = defaultdict(lambda: defaultdict(float))
        self.level1_cnt = defaultdict(int)

        self.level2_sum = defaultdict(lambda: defaultdict(float))
        self.level2_cnt = defaultdict(int)

        self.all_range_sum = defaultdict(float)

        self.level1_range_sum = defaultdict(lambda: defaultdict(float))
        self.level2_range_sum = defaultdict(lambda: defaultdict(float))


        self.detail_list = []


    def _parse_groups(self, fname):
        fname_clean = fname.replace(".csv", "")
        parts = fname_clean.split("_")


        if len(parts) >= 2:
            g1 = parts[1]
        else:
            g1 = parts[0] if parts else "unknown"

  
        if len(parts) >= 5:
            sub_category = parts[4]
            g2 = f"{g1}_{sub_category}"
        else:
            g2 = fname_clean

        return g1, g2

    def _update_sum(self, sum_dict, val_dict):
      
        if not val_dict:
            return
        for k, v in val_dict.items():
            sum_dict[k] += v

    def _get_mean(self, sum_dict, cnt):
      
        if cnt <= 0:
            return {}
        return {k: round(float(sum_dict[k] / cnt), 4) for k in sum_dict}


    def __call__(self, new_result):
        fname = new_result["file_name"]
        p = new_result["point_metrics"]
        r = new_result.get("range_metrics", {})  # 🔥 安全获取 range_metrics
        g1, g2 = self._parse_groups(fname)

        self.detail_list.append(new_result)

      
        self._update_sum(self.all_sum, p)
        self._update_sum(self.all_range_sum, r)
        self.all_cnt += 1


        self._update_sum(self.level1_sum[g1], p)
        self._update_sum(self.level1_range_sum[g1], r)
        self.level1_cnt[g1] += 1


        self._update_sum(self.level2_sum[g2], p)
        self._update_sum(self.level2_range_sum[g2], r)
        self.level2_cnt[g2] += 1

  
        all_list = [{
            "point_metrics": self._get_mean(self.all_sum, self.all_cnt),
            "range_metrics": self._get_mean(self.all_range_sum, self.all_cnt),  # 🔥
            "file_name": "all"
        }]

        level1_list = [{
            "point_metrics": self._get_mean(self.level1_sum[k], self.level1_cnt[k]),
            "range_metrics": self._get_mean(self.level1_range_sum[k], self.level1_cnt[k]),  # 🔥
            "file_name": k
        } for k in self.level1_cnt]

        level2_list = [{
            "point_metrics": self._get_mean(self.level2_sum[k], self.level2_cnt[k]),
            "range_metrics": self._get_mean(self.level2_range_sum[k], self.level2_cnt[k]),  # 🔥
            "file_name": k
        } for k in self.level2_cnt]

        return all_list, level1_list, level2_list, self.detail_list