import numpy as np
from aml_storage import Labels, Block, Descriptor


class DescriptorBuilder:
    def __init__(self, sparse_names, sample_names, component_names, feature_names):
        self._sparse_names = sparse_names
        self.blocks = {}

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
        self.blocks[tuple(sparse)] = block
        return block

    def add_block(
        self, sparse, gradient_samples=None, *, samples=None, components, features=None
    ):
        if samples is None and features is None:
            raise Exception("can not have both samples & features unset")

        if samples is not None and features is not None:
            raise Exception("can not have both samples & features set")

        if samples is not None:
            if isinstance(samples, Labels):
                samples = samples.view(dtype=np.int32).reshape(samples.shape[0], -1)
            samples = Labels(self._sample_names, samples)

        if gradient_samples is not None:
            if not isinstance(gradient_samples, Labels):
                raise Exception("must pass gradient samples for the moment")

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
            block = BlockBuilderPerSamples(
                features, components, self._sample_names, gradient_samples
            )

        if samples is not None:
            block = BlockBuilderPerFeatures(
                samples, components, self._feature_names, gradient_samples
            )

        self.blocks[sparse] = block
        return block

    def build(self):
        sparse = Labels(
            self._sparse_names,
            np.array(list(self.blocks.keys()), dtype=np.int32),
        )

        blocks = []
        for block in self.blocks.values():
            if isinstance(block, Block):
                blocks.append(block)
            elif isinstance(block, BlockBuilderPerFeatures):
                blocks.append(block.build())
            elif isinstance(block, BlockBuilderPerSamples):
                blocks.append(block.build())
            else:
                Exception("Invalid block type")


        self.blocks = {}

        return Descriptor(sparse, blocks)


class BlockBuilderPerSamples:
    def __init__(self, features, components, sample_names, gradient_samples=None):
        assert isinstance(features, Labels)
        assert isinstance(components, Labels)
        assert (gradient_samples is None) or isinstance(gradient_samples, Labels)
        self._gradient_samples = gradient_samples
        self._features = features
        self._components = components

        self._sample_names = sample_names
        self._samples = []

        self._data = []
        self._gradient_data = []

    def add_samples(self, labels, data, gradient=None):
        assert isinstance(data, np.ndarray)
        assert data.shape[2] == self._features.shape[0]
        assert data.shape[1] == self._components.shape[0]

        labels = np.asarray(labels, dtype=np.int32)
        print("wtf", data.shape, len(data.shape))
        if len(data.shape) == 2:
            data = data.reshape(1, data.shape[0], data.shape[1])
            print("wtf squared", data.shape, len(data.shape))
        assert data.shape[0] == labels.shape[0]

        self._samples.append(labels)
        self._data.append(data)

        if gradient is not None:
            raise(Exception("Gradient data not implemented for BlockBuilderSamples"))
            if len(gradient.shape) == 2:
                gradient = gradient.reshape(gradient.shape[0], gradient.shape[1], 1)

            assert gradient.shape[2] == labels.shape[0]
            self._gradient_data.append(gradient)

    def build(self):
        samples = Labels(self._sample_names, np.vstack(self._samples))
        block = Block(
            values=np.concatenate(self._data, axis=0),
            samples=samples,
            components=self._components,
            features=self._features,
        )

        if self._gradient_samples is not None:
            raise(Exception("Gradient data not implemented for BlockBuilderSamples"))
            block.add_gradient(
                "positions",
                self._gradient_samples,
                np.concatenate(self._gradient_data, axis=2),
            )

        self._gradient_data = []
        self._data = []
        self._features = []

        return block


class BlockBuilderPerFeatures:
    def __init__(self, samples, components, feature_names, gradient_samples=None):
        assert isinstance(samples, Labels)
        assert isinstance(components, Labels)
        assert (gradient_samples is None) or isinstance(gradient_samples, Labels)
        self._gradient_samples = gradient_samples
        self._samples = samples
        self._components = components

        self._feature_names = feature_names
        self._features = []

        self._data = []
        self._gradient_data = []

    def add_features(self, labels, data, gradient=None):
        assert isinstance(data, np.ndarray)
        assert data.shape[0] == self._samples.shape[0]
        assert data.shape[1] == self._components.shape[0]

        labels = np.asarray(labels)
        if len(data.shape) == 2:
            data = data.reshape(data.shape[0], data.shape[1], 1)
        assert data.shape[2] == labels.shape[0]

        self._features.append(labels)
        self._data.append(data)

        if gradient is not None:
            if len(gradient.shape) == 2:
                gradient = gradient.reshape(gradient.shape[0], gradient.shape[1], 1)

            assert gradient.shape[2] == labels.shape[0]
            self._gradient_data.append(gradient)

    def build(self):
        features = Labels(self._feature_names, np.vstack(self._features))
        block = Block(
            values=np.concatenate(self._data, axis=2),
            samples=self._samples,
            components=self._components,
            features=features,
        )

        if self._gradient_samples is not None:
            block.add_gradient(
                "positions",
                self._gradient_samples,
                np.concatenate(self._gradient_data, axis=2),
            )

        self._gradient_data = []
        self._data = []
        self._features = []

        return block
