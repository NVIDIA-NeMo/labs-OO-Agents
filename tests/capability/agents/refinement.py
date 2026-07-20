# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Refinement Test Agent. Uses refinement to get the ingredients for a recipe.

Not all ingredients are available in the stock. The agent must find alternative ingredients if the ingredient is not in stock.
- 110 g butter is needed, but only 100 g is in stock.
  The agent must find must find an alternative for the missing 10 g.
  The alternative is available in the needed quantity, therefore the agent can use the alternative or combine multiple alternatives to get the needed quantity.
- 175 g sugar is needed, but only 100 g is in stock.
  The agent must find must find an alternative for the missing 75 g.
  Also the alternative is not available in the needed quantity, therefore the agent needs to combine multiple alternatives to get the needed quantity.
"""

import json  # noqa: F401

from nooa import Agent, CodeActStrategy, strategy
from nooa.config import CodeActConfig


class RefinementTestAgent(Agent):
    """You are an agent that must order the ingredients for a chocolate chip cookie recipe.

    The recipe lists the ingredients and the quantities as follows:
    - 110 g	"butter"
    - 175 g "sugar"
    - 1 packet of "vanilla sugar"
    - 1 teaspoon of "salt"
    - 1 "eggs"
    - 150 g "dark chocolate"
    - 160 g "flour"
    - 1 packet of "baking powder"

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._stock = {
            "butter": 100,
            "sugar": 100,
            "vanilla sugar": 5,
            "salt": 1000,
            "eggs": 12,
            "dark chocolate": 300,
            "flour": 2000,
            "baking powder": 10,
            "coconut oil": 250,
            "honey": 100,
            "maple syrup": 100,
        }
        self._alternatives = {
            "butter": ["margarine", "coconut oil"],
            "sugar": ["honey", "maple syrup"],
            "margarine": ["butter", "coconut oil"],
            "honey": ["sugar", "maple syrup"],
            "coconut oil": ["margarine", "butter"],
            "maple syrup": ["honey", "sugar"],
        }

    async def _order_recipe_ingredients_with_result_parsing(self) -> dict[str, int]:
        result = await self.order_recipe_ingredients()
        corrected_result = await self.place_order(result)
        corrected_result["fatty_group"] = (
            corrected_result.get("butter", 0)
            + corrected_result.get("margarine", 0)
            + corrected_result.get("coconut oil", 0)
        )
        corrected_result["sugary_group"] = (
            corrected_result.get("sugar", 0)
            + corrected_result.get("honey", 0)
            + corrected_result.get("maple syrup", 0)
        )
        corrected_result.pop("butter", None)
        corrected_result.pop("margarine", None)
        corrected_result.pop("coconut oil", None)
        corrected_result.pop("honey", None)
        corrected_result.pop("maple syrup", None)
        corrected_result.pop("sugar", None)
        return corrected_result

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=20)))
    async def order_recipe_ingredients(self) -> dict[str, int]:
        """Order the ingredients for the chocolate chip cookie recipe.

        Exactly the quantities are needed, but substitutions, complements or combinations for an ingredient are allowed.

        For the names, use them as they are given (e.g.: don't rewrite "vanilla sugar" to "vanilla_sugar" nor "eggs" to "egg").
        For the quantity, omit the unit (e.g.: "100 g" becomes the integer value "100").

        Return the result of the place_order function.
        """
        ...

    async def check_availability(self, ingredients: dict[str, int]) -> dict[str, int]:
        """Given the ingredients dictionary, this function checks if the ingredients are in stock.

        For the ingredients in stock, return the ingredients with the requested quantity.
        For the ingredients in stock, but not in the requested quantity, return the ingredients with the available quantity.
        For the ingredients that are not in stock, return 0 quantity.
        """
        return {
            ingredient: min(quantity, self._stock.get(ingredient, 0))
            for ingredient, quantity in ingredients.items()
        }

    async def place_order(self, ingredients: dict[str, int]) -> dict[str, int]:
        """Given the ingredients dictionary, this function places an order for the ingredients.

        If one ingredient is not in stock or not in the requested quantity, this function throws an error.
        """
        error = ""
        for ingredient, quantity in ingredients.items():
            if ingredient not in self._stock or self._stock[ingredient] < quantity:
                error += f"Ingredient {ingredient} is not in stock or not in the requested quantity ({self._stock.get(ingredient, 0)} available)\n"
        if error:
            raise ValueError(error)
        return ingredients

    async def find_alternative(self, ingredient: str) -> list[str]:
        """Given an ingredient, this function finds possilble alternative ingredients that might be used to replace or complement the ingredient in the recipe.
        If the ingredient is not found in the alternatives, an empty list is returned.
        """
        return self._alternatives.get(ingredient, [])
