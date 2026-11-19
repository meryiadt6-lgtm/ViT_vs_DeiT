import React, { useMemo } from 'react';
import katex from 'katex';

interface MathProps {
  math: string;
  className?: string;
}

/**
 * Inline LaTeX formula renderer using KaTeX
 */
export const InlineMath: React.FC<MathProps> = ({ math, className = '' }) => {
  const html = useMemo(() => {
    try {
      return katex.renderToString(math, {
        displayMode: false,
        throwOnError: false,
        output: 'html',
      });
    } catch {
      return math;
    }
  }, [math]);

  return (
    <span
      className={`katex-inline ${className}`}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
};

/**
 * Display / Block LaTeX formula renderer using KaTeX
 */
export const BlockMath: React.FC<MathProps> = ({ math, className = '' }) => {
  const html = useMemo(() => {
    try {
      return katex.renderToString(math, {
        displayMode: true,
        throwOnError: false,
        output: 'html',
      });
    } catch {
      return math;
    }
  }, [math]);

  return (
    <div
      className={`katex-block my-2 overflow-x-auto py-1 text-center ${className}`}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
};

/**
 * Smart Text + LaTeX renderer
 * Automatically turns any $...$ into InlineMath and $$...$$ into BlockMath
 */
export const TeX: React.FC<{ children: string; className?: string }> = ({
  children,
  className = '',
}) => {
  const segments = useMemo(() => {
    if (!children) return [];
    // Split by $$...$$ or $...$
    const parts = children.split(/(\$\$[\s\S]+?\$\$|\$[^$\n]+?\$)/g);
    return parts.map((part, index) => {
      if (part.startsWith('$$') && part.endsWith('$$')) {
        const math = part.slice(2, -2).trim();
        return <BlockMath key={index} math={math} />;
      }
      if (part.startsWith('$') && part.endsWith('$')) {
        const math = part.slice(1, -1).trim();
        return <InlineMath key={index} math={math} />;
      }
      return <React.Fragment key={index}>{part}</React.Fragment>;
    });
  }, [children]);

  return <span className={className}>{segments}</span>;
};

export default TeX;

