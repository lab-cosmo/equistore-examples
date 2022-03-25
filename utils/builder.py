import numpy as np
from aml_storage import Labels, Block, Descriptor


class DescriptorBuilder:
    def __init__(self, sparse_names, sample_names, component_names, feature_names):
        self._sparse_names = sparse_names
        self._sparse = []
        self._blocks = []

        self._sample_names = sample_names
        self._component_names = component_names
        self._feature_names = feature_names

    def add_full_block(self, sparse, samples, components, features, data):
        if isinstance(samples, Labels):
            samples = samples.view(dtype=np.int32).reshape(samples.shape[0], -1)
        samples = Labels(self._sample_names, samples)

        if isinstance(components, Labels):
            components = components.view(dtype=np.int32).reshape(
                components.shape[0], -1
            )
        components = Labels(self._component_names, components)

        if isinstance(features, Labels):
            features = features.view(dtype=np.int32).reshape(features.shape[0], -1)
        features = Labels(self._feature_names, features)

        block = Block(data, samples, components, features)
        self._sparse.append(sparse)
        self._blocks.append(block)
        return block

    def add_block(self, sparse, *, samples=None, components, features=None):
        if samples is None and features is None:
            raise Exception("can not have both samples & features unset")

        if samples is not None and features is not None:
            raise Exception("can not have both samples & features set")

        if samples is not None:
            if isinstance(samples, Labels):
                samples = samples.view(dtype=np.int32).reshape(samples.shape[0], -1)
            samples = Labels(self._sample_names, samples)

        if isinstance(components, Labels):
            components = components.view(dtype=np.int32).reshape(
                components.shape[0], -1
            )
        components = Labels(self._component_names, components)

        if features is not None:
            if isinstance(features, Labels):
                features = features.view(dtype=np.int32).reshape(features.shape[0], -1)
            features = Labels(self._feature_names, features)

        if features is not None:
            raise Exception("not implemented")

        if samples is not None:
            block = BlockBuilderPerFeatures(samples, components, self._feature_names)

        self._sparse.append(sparse)
        self._blocks.append(block)
        return block

    def build(self):
        sparse = Labels(self._sparse_names, np.array(self._sparse, dtype=np.int32))
        self._sparse = []

        blocks = []
        for block in self._blocks:
            if isinstance(block, Block):
                blocks.append(block)
            else:
                assert isinstance(block, BlockBuilderPerFeatures)
                blocks.append(block.build())

        self._blocks = []

        return Descriptor(sparse, blocks)


class BlockBuilderPerFeatures:
    def __init__(self, samples, components, feature_names):
        assert isinstance(samples, Labels)
        assert isinstance(components, Labels)
        self._samples = samples
        self._components = components

        self._feature_names = feature_names
        self._features = []

        self._data = []

    def add_features(self, labels, data):
        assert isinstance(data, np.ndarray)
        assert data.shape[0] == self._samples.shape[0]
        assert data.shape[1] == self._components.shape[0]

        labels = np.asarray(labels)
        if len(data.shape) == 2:
            data = data.reshape(data.shape[0], data.shape[1], 1)
        assert data.shape[2] == labels.shape[0]

        self._features.append(labels)
        self._data.append(data)

    def build(self):
        features = Labels(self._feature_names, np.vstack(self._features))
        block = Block(
            values=np.concatenate(self._data, axis=2),
            samples=self._samples,
            components=self._components,
            features=features,
        )

        self._data = []
        self._features = []

        return block
