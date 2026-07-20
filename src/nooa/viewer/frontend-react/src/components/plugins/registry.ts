import type { TraceEvent, ViewState } from '@/api/types';
import type { ComponentType, ReactNode } from 'react';

export interface PluginProps {
  event: TraceEvent;
  viewState: ViewState;
  searchQuery?: string;
  rawJsonOpen?: boolean;
  viewControls?: ReactNode;
}

export type PluginComponent = ComponentType<PluginProps>;

const exactRegistry = new Map<string, PluginComponent>();
const patternRegistry: { pattern: RegExp; component: PluginComponent }[] = [];

export function registerPlugin(eventType: string, component: PluginComponent) {
  if (eventType.includes('*')) {
    const regex = new RegExp('^' + eventType.replace(/\./g, '\\.').replace(/\*/g, '.*') + '$');
    patternRegistry.push({ pattern: regex, component });
  } else {
    exactRegistry.set(eventType, component);
  }
}

export function getPlugin(eventType: string): PluginComponent | undefined {
  const exact = exactRegistry.get(eventType);
  if (exact) return exact;

  // Prefix match: "span.method.classify" -> try "span.method" -> try "span"
  let prefix = eventType;
  while (prefix.includes('.')) {
    prefix = prefix.substring(0, prefix.lastIndexOf('.'));
    const prefixMatch = exactRegistry.get(prefix);
    if (prefixMatch) return prefixMatch;
  }

  for (const { pattern, component } of patternRegistry) {
    if (pattern.test(eventType)) return component;
  }

  return undefined;
}
