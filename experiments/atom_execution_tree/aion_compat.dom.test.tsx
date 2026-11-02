// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Copy into the pinned Aion fork tests/unit/executionTree directory.
import React from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { IMessageAcpToolCall } from '@/common/chat/chatLib';
import en from '@/renderer/services/i18n/locales/en-US/acp.json';
import ExecutionTree from '@/renderer/pages/conversation/Messages/acp/ExecutionTree';

const { getMessage } = vi.hoisted(() => ({ getMessage: vi.fn() }));
vi.mock('@/common', () => ({ ipcBridge: { database: { getConversationMessage: { invoke: getMessage } } } }));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, string | number>) => {
      const template = en.executionTree[key.split('.').at(-1) as keyof typeof en.executionTree] ?? key;
      return template.replace(/\{\{(\w+)\}\}/g, (_match, name: string) => String(values?.[name] ?? name));
    },
    i18n: { language: 'en-US' },
  }),
}));

// The existing diff card has its own rendering tests; inspect the exact data
// handed to it without loading Electron's preview provider in this DOM test.
vi.mock('@/renderer/pages/conversation/Messages/acp/MessageAcpToolCall', () => ({
  default: ({ message }: { message: IMessageAcpToolCall }) => (
    <pre data-testid='diff-card'>{JSON.stringify(message.content.update.content)}</pre>
  ),
}));


import fs from 'node:fs';
import { composeMessage } from '@/common/chat/chatLib';
import { buildExecutionForest, groupExecutionMessages } from '@/renderer/pages/conversation/Messages/acp/ExecutionTree/executionTree';
afterEach(cleanup);
it('renders actual Atom router/worker observations with inspectable handled failure', async()=>{
 let messages:any[]=[];
 for(const wire of JSON.parse(fs.readFileSync(process.env.ATOM_TREE_CAPTURE ?? '/tmp/atom-tree-smoke/tree-0.json','utf8'))){
  if(!['tool_call','tool_call_update'].includes(wire.sessionUpdate)) continue;
  const {toolCallId,_meta,...update}=wire;
  messages=composeMessage({id:toolCallId,conversation_id:'atom',type:'acp_tool_call',created_at:messages.length,
   position:'left',content:{session_id:'atom',...(_meta?{_meta}:{}),update:{...update,tool_call_id:toolCallId}}},messages);
 }
 const forest=buildExecutionForest(messages);
 expect(forest).toHaveLength(1);
 expect(forest[0].message.content.update.status).toBe('completed');
 expect(groupExecutionMessages(messages).filter(m=>m.type==='execution_tree')).toHaveLength(1);
 render(<ExecutionTree messages={messages}/>);
 expect(screen.getByText('Failed')).toBeInTheDocument();
 const user=userEvent.setup();
 await user.click(screen.getAllByRole('button',{name:'Show details for DemoVerifier.check'})[0]);
 expect(screen.getByText('ValueError: Expected 11, got 10')).toBeInTheDocument();
 const before=screen.getAllByRole('button',{name:/(Show|Hide) details for/}).length;
 expect(before).toBe(13);
 await user.click(screen.getByRole('button',{name:'Collapse all'}));
 expect(screen.getAllByRole('button',{name:/(Show|Hide) details for/})).toHaveLength(1);
 await user.click(screen.getByRole('button',{name:'Expand all'}));
 expect(screen.getAllByRole('button',{name:/(Show|Hide) details for/})).toHaveLength(13);
});
