# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for the specific language governing permissions and limitations under the License.

from herring.torch.parallel import DistributedDataParallel as DDP

class MMDistributedDataParallelHerring(DDP):
    def __init__(self, module, device_ids=None,
                 output_device=None, broadcast_buffers=False,
                 process_group=None, bucket_cap_mb=None):
        super(DistributedDataParallel, self).__init__(module, device_ids, bucket_cap_mb, find_unused_parameters)


    def train_step(self, *inputs, **kwargs):
        result = self.module.train_step(*inputs[0], **kwargs[0])
        self._final_callback_registered = False
        return result

    def val_step(self, *inputs, **kwargs):
        result = self.module.val_step(*inputs[0], **kwargs[0])
        self._final_callback_registered = False
        return result
