# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import pandas as pd

csv_files = [f for f in os.listdir('./copy_csv/') if f.endswith('.csv')]

result_df = pd.DataFrame(columns=['File Name', 'Task Duration(us)'])

for file_name in csv_files:
    file_path = os.path.join('./copy_csv/', file_name)
    df = pd.read_csv(file_path)

    if 'Task Duration(us)' in df.columns:
        durations = df['Task Duration(us)'].tolist()

    temp_df = pd.DataFrame({'File Name': [file_name] * len(durations), 'Task Duration(us)': durations})

    result_df = pd.concat([result_df, temp_df], ignore_index=True)

result_df.to_csv('./all_task_durations.csv', index=False)

print("Save result to all_task_durations.csv.")
