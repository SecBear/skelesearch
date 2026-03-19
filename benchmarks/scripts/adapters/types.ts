/**
 * Adapter contract for benchmark runners.
 *
 * Each adapter encapsulates the tool-specific pipeline: indexing, querying,
 * and artifact assembly. The orchestrator (run-eval.ts) handles path
 * resolution, argument parsing, and writing the output file.
 */

export interface BenchmarkRunInput {
  binary: string;
  repoId: string;
  repoPath: string;
  profilePath: string;
  evalPath: string;
  outputPath: string; // passed through for summary display; orchestrator writes the file
  provider: string;
  reuseIndex: boolean;
}

export interface BenchmarkRunResult {
  /** The fully-assembled artifact object, ready for JSON serialisation. */
  artifact: unknown;
  /** Human-readable one-line summary to print after a successful run. */
  summary: string;
}

export interface BenchmarkAdapter {
  /** Short identifier, used for routing and artifact.tool field. */
  id: string;
  run(input: BenchmarkRunInput): Promise<BenchmarkRunResult>;
}
