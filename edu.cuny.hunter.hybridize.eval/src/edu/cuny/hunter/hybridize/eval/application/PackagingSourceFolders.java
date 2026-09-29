package edu.cuny.hunter.hybridize.eval.application;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Reads the source folders a Python project's packaging metadata declares, for creating a PyDev project from scratch when the evaluator is
 * asked to (see {@link EvaluateHybridizeFunctionRefactoringApplication}). The metadata is what the developer already wrote down about where
 * the importable code lives: {@code setup.cfg}, {@code pyproject.toml}, or {@code setup.py}.
 * <p>
 * The result is the project root, from which a project's own scripts, tests, and examples are run, followed by each package directory the
 * metadata declares, such as {@code src} for a src layout. A flat layout, which declares no package directory, yields the root alone. A
 * project with none of the three files yields nothing, since there is then nothing the developer wrote to go on.
 * <p>
 * The files are read by pattern rather than parsed, so only the common spellings are recognized: setuptools' {@code package_dir} and
 * {@code where}, in all three files, and Poetry's {@code from}. Commented-out lines are ignored, and a declared directory counts only if it
 * exists. A declaration the patterns miss, such as Hatch's or PDM's, yields the root alone; the caller logs the folders it uses, so a miss
 * shows there. This is why reading them is opt-in.
 */
final class PackagingSourceFolders {

	/** The project root, as a source folder relative to it. */
	static final String ROOT = ".";

	/** The packaging files read, in the order they are read. */
	private static final String[] PACKAGING_FILES = { "setup.cfg", "pyproject.toml", "setup.py" };

	/**
	 * Package directories declared by setuptools and Poetry: {@code package_dir = {"": "src"}} and {@code find_packages(where="src")} in
	 * {@code setup.py}; {@code package_dir = =src} and {@code where = src} in {@code setup.cfg}; {@code package-dir = {"" = "src"}},
	 * {@code where = ["src"]}, and Poetry's {@code from = "src"} in {@code pyproject.toml}.
	 */
	private static final Pattern[] PACKAGE_DIRECTORY_PATTERNS = {
			// setup.py: package_dir={"": "src"}; pyproject.toml: package-dir = {"" = "src"}.
			Pattern.compile("package[_-]dir\\s*=\\s*\\{\\s*(['\"])\\1\\s*[:=]\\s*['\"]([^'\"]+)['\"]"),
			// setup.cfg: package_dir = =src, possibly on the next line.
			Pattern.compile("(?m)^\\s*package_dir\\s*=\\s*(?:\\n\\s*)?=\\s*(\\S+)"),
			// setup.py: find_packages("src"), find_namespace_packages("src").
			Pattern.compile("find_(?:namespace_)?packages\\(\\s*['\"]([^'\"]+)['\"]"),
			// setup.py: find_packages(..., where="src", ...), in any argument position.
			Pattern.compile("find_(?:namespace_)?packages\\([^)]*?\\bwhere\\s*=\\s*['\"]([^'\"]+)['\"]"),
			// setup.cfg: where = src.
			Pattern.compile("(?m)^\\s*where\\s*=\\s*([^\\s'\"\\[,]+)\\s*$"),
			// pyproject.toml: where = ["src", "lib"], each entry.
			Pattern.compile("(?:^\\s*where\\s*=\\s*\\[|\\G\\s*,)\\s*['\"]([^'\"]+)['\"]", Pattern.MULTILINE),
			// pyproject.toml (Poetry): packages = [{ include = "x", from = "src" }], an inline table naming what it includes.
			Pattern.compile("\\{[^}]*\\binclude\\s*=[^}]*\\bfrom\\s*=\\s*['\"]([^'\"]+)['\"]") };

	private PackagingSourceFolders() {
	}

	/**
	 * Returns the source folders the packaging metadata in the given directory declares, relative to it: the project root, then each
	 * declared package directory that exists.
	 *
	 * @param directory The project's directory.
	 * @return The source folders, or empty if the directory has no packaging metadata.
	 * @throws IOException If a packaging file cannot be read.
	 */
	static Optional<List<String>> of(File directory) throws IOException {
		Set<String> folders = new LinkedHashSet<>();
		boolean anyPackagingFile = false;

		for (String name : PACKAGING_FILES) {
			File file = new File(directory, name);

			if (!file.isFile())
				continue;

			anyPackagingFile = true;
			// Decoded leniently, since a packaging file need not be UTF-8, and a comment line dropped, since it declares nothing.
			String text = new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8).replaceAll("(?m)^\\s*[#;].*$", "");

			for (Pattern pattern : PACKAGE_DIRECTORY_PATTERNS) {
				Matcher matcher = pattern.matcher(text);

				while (matcher.find()) {
					String folder = matcher.group(matcher.groupCount()).strip();

					if (!folder.isEmpty() && new File(directory, folder).isDirectory())
						folders.add(folder);
				}
			}
		}

		if (!anyPackagingFile)
			return Optional.empty();

		List<String> ret = new ArrayList<>();
		ret.add(ROOT);
		folders.stream().filter(folder -> !folder.equals(ROOT)).forEach(ret::add);
		return Optional.of(ret);
	}
}
