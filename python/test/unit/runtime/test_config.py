# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from unittest.mock import patch

import pytest

from asc.runtime import config


def test_set_platform_model_unavailable_error_interpolates_soc_version():
    soc = config.Platform.Ascend910B3
    with patch.object(config.rt, "is_available", return_value=False), \
         patch.object(config.rt, "use_model"), \
         patch.object(config.rt, "set_soc_version"):
        with pytest.raises(RuntimeError) as exc_info:
            config.set_platform(config.Backend.Model, soc_version=soc, check=True)

    msg = str(exc_info.value)
    assert f"simulator/{soc.value}/lib" in msg
    assert "{soc_version.value}" not in msg
