# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Feature definition and processing."""

from tensorflow import feature_column as tfc
from six import iteritems

class CLVFeatures(object):
  """Encapulates the features for Estimator models."""

  # Columns
  HEADERS = ['customer_id', 'monetary_dnn', 'monetary_btyd', 'frequency_dnn',
             'frequency_btyd', 'recency', 'T', 'time_between',
             'avg_basket_value', 'avg_basket_size', 'cnt_returns',
             'has_returned', 'frequency_btyd_clipped', 'monetary_btyd_clipped',
             'target_monetary_clipped', 'target_monetary']

  HEADERS_DEFAULT = [[''], [0.0], [0.0], [0],
                     [0], [0], [0], [0.0],
                     [0.0], [0.0], [0],
                     [-1], [0], [0.0],
                     [0.0], [0.0]]

  NUMERICS = {
      'monetary_dnn': [],
      'recency': [],
      'frequency_dnn': [],
      'T': [],
      'time_between': [],
      'avg_basket_value': [],
      'avg_basket_size': [],
      'cnt_returns': []}

  CATEGORICALS_W_LIST = {
      'has_returned': [0, 1]}

  # Columns to cross (name, bucket_size, boundaries)
  # Note that boundaries is None if we have all the values. This will helps
  # using between categorical_column_with_identity vs bucketized_column
  # max(recency)=33383
  # max(frequency) = 300
  # max(monetary) = 3809291.2
  CROSSED = []

  KEY = 'customer_id'

  UNUSED = [KEY, 'monetary_btyd', 'frequency_btyd', 'frequency_btyd_clipped',
            'monetary_btyd_clipped', 'target_monetary_clipped']

  TARGET_NAME = 'target_monetary'

  def __init__(self, ignore_crosses=False, is_dnn=None):
    """Initialize CLVFeatures.

    Args:
        ignore_crosses: Whether to apply crosses or not
        is_dnn: Whether the model is a dnn one or not.
    """
    if not is_dnn:
      return

    self.ignore_crosses = ignore_crosses

    # Initializes features names that will be used.
    (self.headers, self.numerics_names,
     self.categoricals_names) = self._keep_used()

    # Creates the base continuous and categorical features
    self.continuous, self.categorical = self._make_base_features()

    # Creates the crossed features for both wide and deep.
    if not self.ignore_crosses:
      self.crossed_for_wide, self.crossed_for_deep = self._make_crossed()

  def _keep_used(self):
    """Returns only the used headers names.

    Returns:
        used_headers names
    """
    headers = [h for h in self.HEADERS if h not in self.UNUSED]
    numerics_names = {
        k: v for k, v in iteritems(self.NUMERICS)
        if (k not in self.UNUSED) and (k != self.TARGET_NAME)
    }
    categoricals_names = {
        k: v for k, v in iteritems(self.CATEGORICALS_W_LIST)
        if k not in self.UNUSED
    }

    return headers, numerics_names, categoricals_names

  def get_key(self):
    return self.KEY

  def get_used_headers(self, with_key=False, with_target=False):
    """Returns headers that are useful to the model.

    Possibly includes the key and the target.

    Args:
        with_key: include KEY column
        with_target: include target column
    Returns:
        used_headers
    """
    used_headers = [h for h in self.headers if h != self.TARGET_NAME]

    if with_key:
      used_headers.insert(0, self.KEY)

    if with_target:
      used_headers.append(self.TARGET_NAME)

    return used_headers

  def get_defaults(self, headers_names=None, with_key=False):
    """Returns default values based on indexes taken from the headers to keep.

    If key and target are to keep, it is decided in get_used_headers.

    Args:
        headers_names: column header names
        with_key: include KEY column
    Returns:
        default values
    """
    if headers_names is None:
      headers_names = self.get_used_headers(with_key)

    keep_indexes = [self.HEADERS.index(n) for n in headers_names]
    return [self.HEADERS_DEFAULT[i] for i in keep_indexes]

  def get_all_names(self):
    return self.HEADERS

  def get_all_defaults(self):
    return self.HEADERS_DEFAULT

  def get_unused(self):
    return self.UNUSED

  def get_target_name(self):
    return self.TARGET_NAME

  #####################
  # Features creation #
  #####################
  # dense columns = numeric columns + embedding columns
  # categorical columns = vocabolary list columns + bucketized columns
  # sparse columns = hashed categorical columns + crossed columns
  # categorical columns => indicator columns
  # deep columns = dense columns + indicator columns
  # wide columns = categorical columns + sparse columns

  def _make_base_features(self):
    """Make base features.

    Returns:
      base features
    """
    # Continuous columns
    continuous = {key_name: tfc.numeric_column(key_name)
                  for key_name in self.numerics_names.keys()}

    # Categorical columns (can contain all categorical_column_with_*)
    categorical = {
        key_name: tfc.categorical_column_with_vocabulary_list(
            key=key_name,
            vocabulary_list=voc)
        for key_name, voc in self.categoricals_names.items()
    }

    return continuous, categorical

  def get_base_features(self):
    # Could create bucket or/and hash here before return
    return self.continuous, self.categorical

  def _prepare_for_crossing(self, key_name, num_bck, boundaries):
    """Prepares features for crossing.

    Whether they're continuous or categorical matters, and
    whether we have the whole dictionary or not.

    Args:
      key_name: A string representing the name of the feature
      num_bck: How many buckets to use when we know # of distinct values
      boundaries: Range used for boundaries when bucketinizing
    Returns:
      key name
    """
    key = None
    if key_name in self.continuous.keys():
      if boundaries is not None:
        # Note that cont[key_name] is a source column
        key = tfc.bucketized_column(self.continuous[key_name], boundaries)
      else:
        # We can count all the values in the dataset. Ex: boolean.
        # Note that key_name is a string
        key = tfc.categorical_column_with_identity(key_name, num_bck)
    elif key_name in self.categorical.keys():
      # It is also possible to use the categorical column instead of the
      # column name. i.e key = cat[key_name]
      key = key_name
    else:
      key = key_name

    return key

  def _make_crossed(self):
    """Makes crossed features for both Wide or Deep network.

    Returns:
      Tuple (crossed columns for Wide, its dimension)
    """
    # Crossed columns
    f_crossed_for_wide = []
    f_crossed_for_deep = []
    for to_cross in self.CROSSED:
      keys = []
      bck_size = 1
      for (key, bck, bnd) in to_cross:
        keys.append(self._prepare_for_crossing(key, bck, bnd))
        bck_size *= bck

      # We can't go crazy on the dim for crossed_column so use a min
      # **0.25 is a rule of thumb for bucket size vs dimension
      t_crossed = tfc.crossed_column(keys, min(bck_size, 10000))
      t_dimension = int(bck_size**0.25)
      f_crossed_for_wide.append(t_crossed)
      f_crossed_for_deep.append(tfc.embedding_column(t_crossed, t_dimension))

    return f_crossed_for_wide, f_crossed_for_deep

  def get_wide_features(self):
    """Creates wide features.

    Sparse (ie. hashed categorical + crossed) + categorical.

    Returns:
      A list of wide features
    """
    # Base sparse (ie categorical) feature columns + crossed
    wide_features = self.categorical.values()

    if not self.ignore_crosses:
      wide_features += self.crossed_for_wide

    return  wide_features

  def get_deep_features(self, with_continuous=True):
    """Creates deep features: dense(ie numeric + embedding) + indicator.

    Args:
      with_continuous: include continuous columns
    Returns:
        features for DNN
    """
    # Multi-hot representation of categories. We know all the values so use
    # indicator_column. If the vocabulary could be bigger in the outside
    # world, we'd use embedding_column
    deep_features = [tfc.indicator_column(f) for f in self.categorical.values()]

    # Creates deep feature lists
    if with_continuous:
      deep_features += self.continuous.values()

    if not self.ignore_crosses:
      deep_features += self.crossed_for_deep

    return deep_features
