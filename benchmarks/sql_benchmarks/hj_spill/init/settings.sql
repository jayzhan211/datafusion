-- Session settings for the hj_spill suite. The memory limit is not set here:
-- `SET datafusion.runtime.memory_limit` always makes a greedy pool, under which
-- the sort-merge baseline runs out of memory (its merge cannot spill,
-- https://github.com/apache/datafusion/issues/14143). The harness sets the
-- limit instead (--memory-limit or MEMORY_LIMIT), with the fair pool by default.
set datafusion.catalog.information_schema = true;
set datafusion.execution.target_partitions = 4;
