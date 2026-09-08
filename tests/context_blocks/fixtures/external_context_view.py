"""A context system implemented using only NOOA's public view contract."""

from nooa import Block


class ExternalContextView:
    async def assemble(self, owner, call):
        yield Block(
            key="external",
            content=f"external context for {call.method_name} on {call.model}",
        )
