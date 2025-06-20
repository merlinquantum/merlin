# MIT License
#
# Copyright (c) 2025 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import json

@dataclass
class Normalization:
    method: str
    range: tuple
    per_feature: bool = True

    def to_text(self) -> str:
        text = f"- Method: {self.method}\n"
        text += f"- Target range: {self.range[0]} to {self.range[1]}\n"
        text += f"- Applied per feature: {self.per_feature}"
        return text

@dataclass
class FeatureNormalization:
    original_unit: Optional[str] = None
    scale_factor: Optional[float] = None
    offset: Optional[float] = None

    def to_text(self) -> str:
        parts = []
        if self.original_unit:
            parts.append(f"original unit: {self.original_unit}")
        if self.scale_factor is not None:
            parts.append(f"scale factor: {self.scale_factor:.3f}")
        if self.offset is not None:
            parts.append(f"offset: {self.offset:.3f}")
        return f"[{', '.join(parts)}]" if parts else ""

@dataclass
class Feature:
    name: str
    description: str
    type: str
    value_range: Optional[tuple] = None
    unit: Optional[str] = None
    stats: Optional[Dict[str, float]] = None
    normalization: Optional[FeatureNormalization] = None

    def to_text(self) -> str:
        text = f"- {self.name} ({self.type}): {self.description}"
        if self.unit:
            text += f" [measured in {self.unit}]"
        if self.value_range:
            text += f" [range: {self.value_range[0]} to {self.value_range[1]}]"
        if self.stats:
            # Separate original and normalized stats
            orig_stats = {k: v for k, v in self.stats.items() if not k.startswith('normalized_')}
            norm_stats = {k.replace('normalized_', ''): v for k, v in self.stats.items() if k.startswith('normalized_')}

            if orig_stats:
                orig_stats_str = ", ".join(f"{k}: {v}" for k, v in orig_stats.items())
                text += f"\n  Original statistics: {orig_stats_str}"
            if norm_stats:
                norm_stats_str = ", ".join(f"{k}: {v}" for k, v in norm_stats.items())
                text += f"\n  Normalized statistics: {norm_stats_str}"

        if self.normalization:
            text += f"\n  Normalization parameters: {self.normalization.to_text()}"
        return text


@dataclass
class DatasetMetadata:
    name: str
    description: str
    features: List[Feature]
    num_instances: int
    subset: str = None
    num_features: int = None
    normalization: Optional[Normalization] = None
    task_type: Optional[List[str]] = field(default_factory=list)
    num_classes: Optional[int] = None
    characteristics: List[str] = field(default_factory=list)
    homepage: Optional[str] = None
    license: Optional[str] = None
    citation: Optional[str] = None
    creators: List[str] = field(default_factory=list)
    year: Optional[int] = None
    feature_relationships: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetMetadata':
        # Extract features from various formats
        features = []
        if 'features' in data:
            for feat_dict in data['features']:
                if isinstance(feat_dict, dict):
                    if 'description' in feat_dict:
                        # Single feature case
                        norm_info = None
                        if 'normalization' in feat_dict:
                            norm_info = FeatureNormalization(**feat_dict['normalization'])

                        features.append(Feature(
                            name=feat_dict['name'],
                            description=feat_dict['description'],
                            type=feat_dict.get('type', 'unknown'),
                            value_range=feat_dict.get('value_range'),
                            unit=feat_dict.get('unit'),
                            stats=feat_dict.get('stats'),
                            normalization=norm_info
                        ))
                    else:
                        # Nested features case
                        for fname, fdata in feat_dict.items():
                            norm_info = None
                            if 'normalization' in fdata:
                                norm_info = FeatureNormalization(**fdata['normalization'])

                            features.append(Feature(
                                name=fname,
                                description=fdata.get('description', ''),
                                type=fdata.get('type', 'unknown'),
                                value_range=fdata.get('value_range'),
                                unit=fdata.get('unit'),
                                stats=fdata.get('stats'),
                                normalization=norm_info
                            ))

        if 'num_features' in data:
            num_features = data['num_features']
        elif len(features):
            num_features = len(features)
        else:
            num_features = None

        # Get normalization info if present
        normalization = None
        if 'normalization' in data:
            normalization = Normalization(**data['normalization'])

        # Get number of instances
        num_instances = data.get('num_instances', 0)
        if not num_instances:
            if 'dataset_characteristics' in data:
                num_instances = data['dataset_characteristics'].get('num_instances', 0)
            elif 'splits' in data:
                num_instances = sum(split.get('num_examples', 0) for split in data['splits'].values())

        return cls(
            name=data['name'],
            subset = data.get('subset'),
            description=data.get('description', data.get('abstract', '')).strip(),
            features=features,
            num_instances=num_instances,
            num_features=num_features,
            task_type=data.get('task', []),
            num_classes=data.get('num_classes'),
            characteristics=data.get('characteristics', []),
            homepage=data.get('homepage'),
            license=data.get('license'),
            citation=data.get('citation'),
            creators=data.get('creators', []),
            year=data.get('year_of_dataset_creation',
                          data.get('year',
                                   datetime.now().year)),
            normalization=normalization
        )

    def __str__(self) -> str:
        """Generate a human-readable text representation of the dataset metadata"""
        text = [
            f"Dataset: {self.name}",
        ]
        if self.subset:
            text.append(f"Subset: {self.subset}")
        text += [
            f"\nDescription:",
            self.description,
            f"\nBasic Information:",
            f"- Number of instances: {self.num_instances:,}",
            f"- Number of features: {self.num_features:,}",
        ]

        if self.task_type:
            text.append(f"- Task types: {', '.join(self.task_type)}")
        if self.num_classes:
            text.append(f"- Number of classes: {self.num_classes}")
        if self.characteristics:
            text.append(f"- Characteristics: {', '.join(self.characteristics)}")

        if self.normalization:
            text.append("\nNormalization Information:")
            text.append(self.normalization.to_text())

        if self.features:
            text.append("\nFeatures:")
            for feature in self.features:
                text.append(feature.to_text())

        if self.feature_relationships:
            text.append("\nFeature Relationships:")
            text.append(self.feature_relationships)

        if self.creators:
            text.append(f"\nCreators: {', '.join(self.creators)}")
        if self.year:
            text.append(f"Year: {self.year}")
        if self.homepage:
            text.append(f"Homepage: {self.homepage}")
        if self.license:
            text.append(f"License: {self.license}")
        if self.citation:
            text.append(f"\nCitation:")
            text.append(self.citation)

        return "\n".join(text)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the metadata to a dictionary format"""
        return {
            "name": self.name,
            "subset": self.subset,
            "description": self.description,
            "features_relationships": self.feature_relationships,
            "features": [
                {
                    "name": f.name,
                    "description": f.description,
                    "type": f.type,
                    "value_range": f.value_range,
                    "unit": f.unit,
                    "stats": f.stats,
                    "normalization": {
                        "original_unit": f.normalization.original_unit,
                        "scale_factor": f.normalization.scale_factor,
                        "offset": f.normalization.offset
                    } if f.normalization else None
                }
                for f in self.features
            ],
            "num_features": self.num_features,
            "num_instances": self.num_instances,
            "task_type": self.task_type,
            "num_classes": self.num_classes,
            "characteristics": self.characteristics,
            "homepage": self.homepage,
            "license": self.license,
            "citation": self.citation,
            "creators": self.creators,
            "year": self.year,
            "normalization": {
                "method": self.normalization.method,
                "range": self.normalization.range,
                "per_feature": self.normalization.per_feature
            } if self.normalization else None
        }


# Example usage:
if __name__ == "__main__":
    from merlin.datasets import iris
    # Example of converting one of the datasets
    _, _, iris_data = iris.get_data_train()
    print(iris_data)