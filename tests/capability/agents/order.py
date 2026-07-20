# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Order/fast-food conversation agent."""

from enum import StrEnum

from pydantic import BaseModel

from nooa import Agent


class ErrorCode(StrEnum):
    """Error codes for order operations."""

    ITEM_NOT_IN_MENU = "ITEM_NOT_IN_MENU"
    ITEM_NOT_IN_ORDER = "ITEM_NOT_IN_ORDER"
    INVALID_SIZE = "INVALID_SIZE"
    INVALID_ADDITION = "INVALID_ADDITION"
    INVALID_REMOVAL = "INVALID_REMOVAL"
    ORDER_EMPTY = "ORDER_EMPTY"


class Modifications(BaseModel):
    """Modifications to an item with additions, removals, and special instructions.

    Args:
        additions: List of ingredients to add (e.g. "cheese", "bacon").
        removals: List of ingredients to remove (e.g. "lettuce", "tomato").
        special_instructions: List of special requests (e.g. "extra cheese", "light onion").
    """

    additions: list[str] | None = None  # ["cheese", "bacon"] or omitted
    removals: list[str] | None = None  # ["lettuce", "tomato"] or omitted
    special_instructions: list[str] | None = None  # ["extra cheese"] or omitted


class OrderItem(BaseModel):
    """Order item with product ID, size, and modifications.

    Args:
        product_id: Product ID from the menu.
        size: Size of the item.
        modifications: Modifications to the item.
    """

    product_id: int
    size: str | None = None  # "small", "medium", "large", etc. or omitted
    modifications: Modifications | None = None


class OrderState(BaseModel):
    """Order state with submitted, canceled, and items.

    Args:
        order_submitted: Whether the order has been submitted.
        order_canceled: Whether the order has been canceled.
        order_items: Dictionary of items in the order, keyed by order_item_id.
    """

    order_submitted: bool | None
    order_canceled: bool | None
    order_items: dict[int, OrderItem] = {}

    def model_dump(self, **kwargs):
        """Serialize order_items as list (without order_item_ids) for simpler test comparison."""
        data = super().model_dump(**kwargs)
        if "order_items" in data:
            data["order_items"] = list(data["order_items"].values())
        return data


class MenuItem(BaseModel):
    """Menu item definition with name, description, and allowed modifications.

    Args:
        name: Display name of the item.
        description: Brief description of the item.
        sizes: List of sizes available for the item.
        default_ingredients: List of ingredients that come with the item by default.
        allowed_additions: List of additions that can be added to the item.
    """

    name: str
    description: str
    sizes: list[str] | None = None
    default_ingredients: list[str] = []
    allowed_additions: list[str] = []


MENU: dict[int, MenuItem] = {
    1001: MenuItem(
        name="burger",
        description="Classic beef patty with fresh vegetables",
        sizes=None,
        default_ingredients=["bun", "patty", "lettuce", "tomato"],
        allowed_additions=["cheese", "bacon", "onion", "pickles"],
    ),
    1002: MenuItem(
        name="chicken sandwich",
        description="Crispy chicken breast with fresh toppings",
        sizes=None,
        default_ingredients=["bun", "chicken", "lettuce", "tomato"],
        allowed_additions=["cheese", "bacon", "pickles", "mayo"],
    ),
    2001: MenuItem(
        name="fries",
        description="Golden crispy french fries",
        sizes=["small", "medium", "large"],
        default_ingredients=["salt"],
        allowed_additions=["cheese sauce", "ranch"],
    ),
    2002: MenuItem(
        name="onion rings",
        description="Crispy battered onion rings",
        sizes=["small", "medium", "large"],
        default_ingredients=["salt"],
        allowed_additions=["ranch", "bbq sauce"],
    ),
    3001: MenuItem(
        name="coke",
        description="Coca-Cola",
        sizes=["small", "medium", "large"],
        default_ingredients=[],
        allowed_additions=["ice", "lemon"],
    ),
    3002: MenuItem(
        name="sprite",
        description="Sprite",
        sizes=["small", "medium", "large"],
        default_ingredients=[],
        allowed_additions=["ice", "lemon"],
    ),
}


class OrderTestWrapper(Agent):
    """Wrapper for fast_food order tests.

    Handles multi-turn conversations and maintains order state.
    Returns the final state for verification.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order_items: dict[int, OrderItem] = {}
        self._next_order_item_id: int = 5000
        self.order_submitted: bool = False
        self.order_canceled: bool = False

    def _validate_item(
        self,
        product_id: int,
        size: str | None = None,
        additions: list[str] | None = None,
        removals: list[str] | None = None,
        allow_clear: bool = False,
    ) -> ErrorCode | MenuItem:
        """Validate item against menu constraints.

        Args:
            product_id: Product ID to validate (4-digit integer)
            size: Size to validate (None = no size, "" = clear size if allow_clear)
            additions: Additions to validate
            removals: Removals to validate
            allow_clear: If True, empty string size and empty lists are allowed (for modify)

        Returns:
            MenuItem if valid, ErrorCode if invalid.
        """
        # Validate item exists in menu
        if product_id not in MENU:
            return ErrorCode.ITEM_NOT_IN_MENU

        menu_item = MENU[product_id]

        # Validate size (skip if clearing with empty string)
        if size is not None and not (allow_clear and size == ""):
            if menu_item.sizes is None:
                return ErrorCode.INVALID_SIZE
            if size not in menu_item.sizes:
                return ErrorCode.INVALID_SIZE

        # Validate additions (skip if clearing with empty list)
        if additions and not (allow_clear and len(additions) == 0):
            for addition in additions:
                if addition not in menu_item.allowed_additions:
                    return ErrorCode.INVALID_ADDITION

        # Validate removals (skip if clearing with empty list)
        if removals and not (allow_clear and len(removals) == 0):
            for removal in removals:
                if removal not in menu_item.default_ingredients:
                    return ErrorCode.INVALID_REMOVAL

        return menu_item

    # Tools available to the agent
    async def add_item(
        self,
        product_id: int,
        size: str | None = None,
        additions: list[str] | None = None,
        removals: list[str] | None = None,
        special_instructions: list[str] | None = None,
    ) -> ErrorCode | int:
        """Add an item to the order with optional size and modifications.

        Args:
            product_id: Product ID from the menu (4-digit integer, e.g., 1001 for burger)
            size: Size of the item - must be valid for item (e.g., "small", "medium", "large").
                  Defaults to "medium" for items with size options.
            additions: Ingredients to add - must be in allowed_additions for item
            removals: Ingredients to remove - must be in default_ingredients for item
            special_instructions: Special requests (free-form). Use for quantity/preparation modifiers (e.g., "extra cheese", "light onion").

        Returns:
            Order item ID on success, or ErrorCode on failure.
        """
        result = self._validate_item(product_id, size, additions, removals)
        if isinstance(result, ErrorCode):
            return result

        menu_item = result

        # Default to medium size if item has sizes and none specified
        if size is None and menu_item.sizes is not None:
            size = "medium"

        # All validations passed - add the item
        mods = None
        if additions or removals or special_instructions:
            mods = Modifications(
                additions=list(additions) if additions else None,
                removals=list(removals) if removals else None,
                special_instructions=list(special_instructions) if special_instructions else None,
            )
        order_item_id = self._next_order_item_id
        self._next_order_item_id += 1
        entry = OrderItem(product_id=product_id, size=size, modifications=mods)
        self.order_items[order_item_id] = entry
        return order_item_id

    async def remove_item(self, order_item_id: int) -> ErrorCode | None:
        """Remove an item from the order by order item ID.

        Args:
            order_item_id: The unique ID of the order item to remove.

        Returns:
            None on success, or ErrorCode.ITEM_NOT_IN_ORDER if item not found.
        """
        if order_item_id not in self.order_items:
            return ErrorCode.ITEM_NOT_IN_ORDER

        del self.order_items[order_item_id]

    async def modify_item(
        self,
        order_item_id: int,
        additions: list[str] | None = None,
        removals: list[str] | None = None,
        special_instructions: list[str] | None = None,
        size: str | None = None,
    ) -> ErrorCode | None:
        """Modify an existing item's modifications or size.

        Args:
            order_item_id: The unique ID of the order item to modify.
            additions: Ingredients to add - must be in allowed_additions (["cheese"]), or [] to clear
            removals: Ingredients to remove - must be in default_ingredients (["lettuce"]), or [] to clear
            special_instructions: Special requests (free-form) or [] to clear. Use for quantity/preparation modifiers (e.g., "extra cheese", "light onion").
            size: New size - must be valid for item (e.g., "large"), or "" to clear

        Returns:
            None on success, or ErrorCode on failure.
        """
        if order_item_id not in self.order_items:
            return ErrorCode.ITEM_NOT_IN_ORDER

        entry = self.order_items[order_item_id]

        # Validate against menu (allow_clear=True for empty lists and empty string size)
        validation = self._validate_item(
            entry.product_id, size, additions, removals, allow_clear=True
        )
        if isinstance(validation, ErrorCode):
            return validation

        # All validations passed - apply modifications

        # Update size if provided (empty string "" clears the size)
        if size is not None:
            entry.size = size if size else None

        # Handle modifications (empty list [] clears, non-empty extends, None = no change)
        if additions is not None or removals is not None or special_instructions is not None:
            if entry.modifications is None:
                entry.modifications = Modifications()

            def extend_unique(target: list, items: list):
                for item in items:
                    if item not in target:
                        target.append(item)

            # Handle additions list: [] clears, ["cheese"] extends (deduped)
            if additions is not None:
                if len(additions) == 0:
                    entry.modifications.additions = None
                else:
                    if entry.modifications.additions is None:
                        entry.modifications.additions = []
                    extend_unique(entry.modifications.additions, additions)

            # Handle removals list: [] clears, ["lettuce"] extends (deduped)
            if removals is not None:
                if len(removals) == 0:
                    entry.modifications.removals = None
                else:
                    if entry.modifications.removals is None:
                        entry.modifications.removals = []
                    extend_unique(entry.modifications.removals, removals)

            # Handle special_instructions list: [] clears, ["extra cheese"] extends (deduped)
            if special_instructions is not None:
                if len(special_instructions) == 0:
                    entry.modifications.special_instructions = None
                else:
                    if entry.modifications.special_instructions is None:
                        entry.modifications.special_instructions = []
                    extend_unique(entry.modifications.special_instructions, special_instructions)

            # Clean up: if all modification fields are None, set modifications to None
            if (
                entry.modifications.additions is None
                and entry.modifications.removals is None
                and entry.modifications.special_instructions is None
            ):
                entry.modifications = None

    async def submit_order(self) -> ErrorCode | None:
        """Submit the final order.

        Returns:
            None on success, or ErrorCode.ORDER_EMPTY if no items in order.
        """
        if not self.order_items:
            return ErrorCode.ORDER_EMPTY

        self.order_submitted = True

    async def cancel_order(self) -> None:
        """Cancel the order."""

        self.order_canceled = True
        self.order_items = {}

    async def get_order_status(self) -> str:
        """Get current order status."""
        items_summary = {
            order_item_id: item.model_dump(exclude_none=True)
            for order_item_id, item in self.order_items.items()
        }
        return f"Current order: {items_summary}"

    async def get_menu(self) -> str:
        """Get the full menu with all product information id, name, description, sizes, default ingredients, and allowed additions.

        Returns:
            Formatted string showing all menu items with IDs, names, descriptions, and options.
        """
        lines = ["=== MENU ==="]
        for product_id, item in MENU.items():
            lines.append(f"\n{item.name.title()} (ID: {product_id})")
            lines.append(f"Description: {item.description}")
            if item.sizes:
                lines.append(f"Sizes: {', '.join(item.sizes)} (default: 'medium')")
            if item.default_ingredients:
                lines.append(f"Comes with: {', '.join(item.default_ingredients)}")
            if item.allowed_additions:
                lines.append(f"Optionally, can add: {', '.join(item.allowed_additions)}")
        return "\n".join(lines)

    async def process_message(self, user_message: str) -> dict:
        """Process one message from a silent stream of customer order updates.

        The customer cannot hear intermediate replies. Do not rely on asking the
        customer clarifying questions unless the message is genuinely impossible to
        map to the menu. For every concrete request, immediately update order state
        by calling the appropriate tools.

        State structure:
        - order_items: dict[int, OrderItem] where key is order_item_id
        - OrderItem: product_id (int), size (str | None), modifications (Modifications | None)
        - Modifications: additions (list[str] | None), removals (list[str] | None), special_instructions (list[str] | None)

        Rules:
        - Track the full order across messages; later messages modify the existing order.
        - Do not duplicate existing items when a message modifies size or toppings.
        - Only call submit_order() when the customer explicitly asks to submit/finalize.
        - Only call cancel_order() when the customer explicitly asks to cancel.
        - When the conversation is complete, the final state must be submitted or canceled.

        Return a dict with:
        - message: brief internal acknowledgement of the state update
        """
        ...

    async def run_conversation(self, messages: list[str]) -> dict:
        """Run a full multi-turn conversation and return final state.

        This is the entry point for the eval_pipeline.
        It processes all messages in sequence and returns the final state.
        """
        for msg in messages:
            await self.process_message(msg)

        # Return full state using Pydantic model
        state = OrderState(
            order_submitted=True if self.order_submitted else None,
            order_canceled=True if self.order_canceled else None,
            order_items=self.order_items,
        )
        return state.model_dump(exclude_none=True)


class FastFoodOrderScorer:
    """Score fast-food order state while ignoring underspecified modifier encoding.

    The order tests care about final state: submitted/canceled flags and ordered items.
    Independent order items are compared order-insensitively, while multiplicity is
    preserved so duplicate or missing items still fail.  For item modifiers,
    ``"extra mayo"`` can reasonably be represented either as ``additions=["mayo"]``
    or as ``special_instructions=["extra mayo"]``. Normalize those quantity-style
    instructions into additions when the item supports the ingredient, so missing
    real items (for example a Coke) still fail but encoding choices do not.
    """

    def score(self, ctx):
        from eval_pipeline.models import ScoreResult

        expected = self._normalize_state(self._to_plain(ctx.expected))
        actual = self._normalize_state(self._to_plain(ctx.actual))
        errors = self._compare(expected, actual)
        return ScoreResult(
            score=0.0 if errors else 1.0,
            reasoning="Order state matches" if not errors else "; ".join(errors[:6]),
            metadata={"errors": errors, "error_count": len(errors)},
        )

    def _to_plain(self, value):
        if hasattr(value, "model_dump"):
            value = value.model_dump(exclude_none=True)
        elif hasattr(value, "dict"):
            value = value.dict(exclude_none=True)
        if isinstance(value, dict):
            return {k: self._to_plain(v) for k, v in value.items() if v is not None}
        if isinstance(value, list):
            return [self._to_plain(v) for v in value]
        return value

    def _normalize_state(self, state):
        if not isinstance(state, dict):
            return state
        normalized = dict(state)
        items = normalized.get("order_items")
        if isinstance(items, dict):
            items = list(items.values())
        if isinstance(items, list):
            normalized["order_items"] = sorted(
                (self._normalize_item(item) for item in items), key=self._item_sort_key
            )
        return normalized

    def _item_sort_key(self, item):
        if not isinstance(item, dict):
            return repr(item)
        modifications = item.get("modifications") or {}
        return (
            item.get("product_id"),
            item.get("size") or "",
            tuple(modifications.get("additions") or []),
            tuple(modifications.get("removals") or []),
            tuple(modifications.get("special_instructions") or []),
        )

    def _normalize_item(self, item):
        if not isinstance(item, dict):
            return item
        normalized = dict(item)
        if "modifications" in normalized:
            mods = self._normalize_modifications(
                normalized.get("product_id"), normalized.get("modifications")
            )
            if mods:
                normalized["modifications"] = mods
            else:
                normalized.pop("modifications", None)
        return {k: v for k, v in normalized.items() if v is not None}

    def _normalize_modifications(self, product_id, modifications):
        if not isinstance(modifications, dict):
            return modifications
        additions = list(modifications.get("additions") or [])
        removals = list(modifications.get("removals") or [])
        special = list(modifications.get("special_instructions") or [])

        menu_item = MENU.get(product_id)
        allowed_additions = set(menu_item.allowed_additions) if menu_item else set()
        remaining_special = []
        for instruction in special:
            normalized = self._normalize_text(instruction)
            if normalized.startswith("extra "):
                ingredient = normalized.removeprefix("extra ").strip()
                if ingredient in allowed_additions:
                    if ingredient not in additions:
                        additions.append(ingredient)
                    continue
            remaining_special.append(instruction)

        result = {}
        if additions:
            result["additions"] = sorted(additions)
        if removals:
            result["removals"] = sorted(removals)
        if remaining_special:
            result["special_instructions"] = sorted(remaining_special)
        return result

    def _normalize_text(self, value):
        return " ".join(str(value).strip().lower().split())

    def _compare(self, expected, actual):
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            return [] if expected == actual else [f"expected {expected!r}, got {actual!r}"]

        errors = []
        for key in ("order_submitted", "order_canceled"):
            if expected.get(key) != actual.get(key):
                errors.append(f"{key}: expected {expected.get(key)!r}, got {actual.get(key)!r}")

        expected_items = expected.get("order_items", [])
        actual_items = actual.get("order_items", [])
        if expected_items != actual_items:
            errors.append(f"order_items: expected {expected_items!r}, got {actual_items!r}")

        extra_keys = set(actual.keys()) - {"order_submitted", "order_canceled", "order_items"}
        if extra_keys:
            errors.append(f"unexpected keys in actual: {sorted(extra_keys)!r}")
        return errors
