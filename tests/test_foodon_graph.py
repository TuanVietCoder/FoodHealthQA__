"""
Unit tests for FoodOn Knowledge Graph structure and Graph Subsumption Reasoning.
Tests:
  1. Bidirectional Graph Indexing (parents & children)
  2. Ancestor DAG Traversal (get_ancestor_curies, get_ancestors)
  3. Descendant DAG Traversal (get_descendant_curies, get_descendants)
  4. Hierarchical Subclass Verification (is_subclass_of)
  5. Lowest Common Ancestor (LCA) Finding (find_common_ancestors, find_lowest_common_ancestor)
  6. Lineage Tracing (get_lineage)
  7. 1-Hop Graph Neighborhood (get_graph_neighborhood)
  8. Graph Subsumption in Constraint Modeling (classify_constraint_type with FoodOn graph)
"""

import os
import unittest

from modules.constraint_model import (
    CONSTRAINT_ALLERGEN_RISK,
    CONSTRAINT_PERMITTED,
    classify_constraint_type,
)
from modules.foodon import (
    find_foodon_common_ancestor,
    get_foodon_ancestors,
    get_foodon_descendants,
    get_foodon_graph_neighborhood,
    get_foodon_lineage,
    is_foodon_subclass_of,
    load_foodon_entities,
)


class TestFoodOnGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        cls.catalog = load_foodon_entities(data_dir)
        cls.peanut_curie = "FOODON:00002099"      # peanut food product
        cls.nut_curie = "FOODON:00001172"         # nut food product
        cls.legume_curie = "FOODON:00001264"      # legume food product
        cls.additive_curie = "FOODON:03412972"    # food additive
        cls.starch_curie = "FOODON:03412981"      # acetylated starch
        cls.glyceride_curie = "FOODON:03413156"   # mono- and diglycerides

    def test_catalog_loaded_with_graph_indices(self):
        self.assertGreater(len(self.catalog), 25000, "Catalog should have > 25,000 entities")
        self.assertGreater(len(self.catalog.parents), 25000, "Parents map should be indexed")
        self.assertGreater(len(self.catalog.children), 5000, "Children map should have > 5,000 parent nodes")

    def test_ancestor_traversal(self):
        ancestors = self.catalog.get_ancestor_curies(self.peanut_curie, max_depth=10)
        self.assertIn(self.nut_curie, ancestors, "Peanut should have nut food product as ancestor")
        self.assertIn(self.legume_curie, ancestors, "Peanut should have legume food product as ancestor")
        self.assertNotIn(self.additive_curie, ancestors, "Peanut should NOT have food additive as ancestor")

    def test_descendant_traversal(self):
        descendants = self.catalog.get_descendant_curies(self.nut_curie, max_depth=10)
        self.assertIn(self.peanut_curie, descendants, "Nut food product should have peanut in descendants")

    def test_is_subclass_of(self):
        # Direct & transitive true relations
        self.assertTrue(self.catalog.is_subclass_of(self.peanut_curie, self.nut_curie))
        self.assertTrue(self.catalog.is_subclass_of(self.peanut_curie, self.legume_curie))
        self.assertTrue(self.catalog.is_subclass_of(self.starch_curie, self.additive_curie))
        # Self-identity
        self.assertTrue(self.catalog.is_subclass_of(self.peanut_curie, self.peanut_curie))
        # False relations
        self.assertFalse(self.catalog.is_subclass_of(self.peanut_curie, self.additive_curie))
        self.assertFalse(self.catalog.is_subclass_of(self.additive_curie, self.peanut_curie))

    def test_common_ancestor_and_lca(self):
        common = self.catalog.find_common_ancestors(self.starch_curie, self.glyceride_curie)
        self.assertIn(self.additive_curie, common, "Both starch and glyceride additives share 'food additive'")

        lca = self.catalog.find_lowest_common_ancestor(self.starch_curie, self.glyceride_curie)
        self.assertIsNotNone(lca)
        self.assertIn(lca["curie"], common)

    def test_lineage(self):
        lineage = self.catalog.get_lineage(self.peanut_curie, max_depth=6)
        self.assertGreaterEqual(len(lineage), 2, "Lineage should have at least 2 nodes")
        self.assertEqual(lineage[-1]["curie"], self.peanut_curie, "Last element of lineage must be leaf entity")

    def test_graph_neighborhood(self):
        nh = self.catalog.get_graph_neighborhood(self.peanut_curie)
        self.assertIsNotNone(nh)
        self.assertEqual(nh["curie"], self.peanut_curie)
        self.assertGreaterEqual(len(nh["parents"]), 1)
        self.assertGreaterEqual(nh["total_children"], 1)

    def test_module_level_helpers(self):
        self.assertTrue(is_foodon_subclass_of(self.peanut_curie, self.nut_curie, self.catalog))
        anc = get_foodon_ancestors(self.peanut_curie, self.catalog)
        self.assertIn(self.nut_curie, anc)
        desc = get_foodon_descendants(self.nut_curie, self.catalog)
        self.assertIn(self.peanut_curie, desc)
        lin = get_foodon_lineage(self.peanut_curie, self.catalog)
        self.assertTrue(len(lin) >= 2)
        lca = find_foodon_common_ancestor(self.starch_curie, self.glyceride_curie, self.catalog)
        self.assertIsNotNone(lca)
        nh = get_foodon_graph_neighborhood(self.peanut_curie, self.catalog)
        self.assertEqual(nh["curie"], self.peanut_curie)

    def test_constraint_model_graph_subsumption(self):
        # Persona with tree nut allergy
        persona = {"conditions": [], "allergies": ["tree_nuts"]}
        # Product item has FoodOn CURIE for peanut (which is a subclass of nut food product)
        additive = {"foodon_curie": self.peanut_curie}
        res = classify_constraint_type("kẹo đậu phộng", additive, persona)
        self.assertEqual(res["constraint_type"], CONSTRAINT_ALLERGEN_RISK)
        self.assertIn("tree_nuts_allergy", res["matched_rule"][0])

        # Persona without allergies should remain permitted
        persona_clean = {"conditions": [], "allergies": []}
        res_clean = classify_constraint_type("kẹo đậu phộng", additive, persona_clean)
        self.assertEqual(res_clean["constraint_type"], CONSTRAINT_PERMITTED)


if __name__ == "__main__":
    unittest.main()

