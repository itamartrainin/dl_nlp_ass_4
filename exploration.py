import json
import re

fnames = ['../data/snli_1.0/snli_1.0/snli_1.0_train.jsonl',
          '../data/snli_1.0/snli_1.0/snli_1.0_dev.jsonl',
          '../data/snli_1.0/snli_1.0/snli_1.0_test.jsonl',
          '../data/multinli_1.0/multinli_1.0/multinli_1.0_train.jsonl',
          '../data/multinli_1.0/multinli_1.0/multinli_1.0_dev_matched.jsonl',
          '../data/multinli_1.0/multinli_1.0/multinli_1.0_dev_mismatched.jsonl']

LHS_total_max_len = 0
RHS_total_max_len = 0

for fname in fnames:
    with open(fname, 'r', encoding='utf-8') as f:

        true_line_counter = 0
        false_line_counter = 0

        LHS_max_len = 0
        LHS_ave_len = 0

        RHS_max_len = 0
        RHS_ave_len = 0

        for line in f.readlines():
            line = json.loads(line)
            if line['gold_label'] != '-':
                true_line_counter += 1
                LHS_token_count = len(list(filter(lambda x: x != '', re.split(r'[ ()]', line['sentence1_binary_parse']))))
                RHS_token_count = len(list(filter(lambda x: x != '', re.split(r'[ ()]', line['sentence2_binary_parse']))))
                LHS_ave_len += LHS_token_count
                RHS_ave_len += RHS_token_count
                if LHS_token_count > LHS_max_len:
                    LHS_max_len = LHS_token_count
                if RHS_token_count > RHS_max_len:
                    RHS_max_len = RHS_token_count
            else:
                false_line_counter += 1

        if LHS_max_len > LHS_total_max_len:
            LHS_total_max_len = LHS_max_len
        if RHS_max_len > RHS_total_max_len:
            RHS_total_max_len = RHS_max_len

    print('\'' + fname + '\'')
    print('true_line_counter: ' + str(true_line_counter))
    print('false_line_counter: ' + str(false_line_counter))
    print('total_line_counter: ' + str(true_line_counter + false_line_counter))
    print('LHS_max_len: ' + str(LHS_max_len))
    print('LHS_ave_len: ' + str(LHS_ave_len / true_line_counter))
    print('RHS_max_len: ' + str(RHS_max_len))
    print('RHS_ave_len: ' + str(RHS_ave_len / true_line_counter))
    print()

print('LHS_total_max_len: ' + str(LHS_total_max_len))
print('RHS_total_max_len: ' + str(RHS_total_max_len))

# ===== OUTPUT =====
# '../data/snli_1.0/snli_1.0/snli_1.0_train.jsonl'
# true_line_counter: 549367
# false_line_counter: 785
# total_line_counter: 550152
# LHS_max_len: 82
# LHS_ave_len: 14.029819774394895
# RHS_max_len: 62
# RHS_ave_len: 8.251828012967652
#
# '../data/snli_1.0/snli_1.0/snli_1.0_dev.jsonl'
# true_line_counter: 9842
# false_line_counter: 158
# total_line_counter: 10000
# LHS_max_len: 59
# LHS_ave_len: 15.18644584434058
# RHS_max_len: 55
# RHS_ave_len: 8.353281853281853
#
# '../data/snli_1.0/snli_1.0/snli_1.0_test.jsonl'
# true_line_counter: 9824
# false_line_counter: 176
# total_line_counter: 10000
# LHS_max_len: 57
# LHS_ave_len: 15.155028501628664
# RHS_max_len: 30
# RHS_ave_len: 8.316571661237784
#
# '../data/multinli_1.0/multinli_1.0/multinli_1.0_train.jsonl'
# true_line_counter: 392702
# false_line_counter: 0
# total_line_counter: 392702
# LHS_max_len: 401
# LHS_ave_len: 22.281898233265938
# RHS_max_len: 70
# RHS_ave_len: 11.326695560501348
#
# '../data/multinli_1.0/multinli_1.0/multinli_1.0_dev_matched.jsonl'
# true_line_counter: 9815
# false_line_counter: 185
# total_line_counter: 10000
# LHS_max_len: 205
# LHS_ave_len: 21.68762098828324
# RHS_max_len: 51
# RHS_ave_len: 11.254610290371879
#
# '../data/multinli_1.0/multinli_1.0/multinli_1.0_dev_mismatched.jsonl'
# true_line_counter: 9832
# false_line_counter: 168
# total_line_counter: 10000
# LHS_max_len: 146
# LHS_ave_len: 22.552583401139138
# RHS_max_len: 65
# RHS_ave_len: 12.183685923515053
#
# LHS_total_max_len: 401
# RHS_total_max_len: 70
