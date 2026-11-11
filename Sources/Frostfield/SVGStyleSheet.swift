import Foundation

/// The class and ID selector subset used by SVGLoader.
struct SVGStyleSheet {
    private var rules: [String: [String: String]] = [:]
    private var remainingSelectors = 4_096

    mutating func add(_ text: String) throws {
        let stripped = Self.withoutComments(text)
        var rest = stripped[...]
        while let boundary = rest.firstIndex(where: { $0 == "{" || $0 == ";" }) {
            let selectors = rest[..<boundary]
            rest = rest[rest.index(after: boundary)...]
            // At-statements such as @import are skipped without looking up a resource.
            if stripped[boundary] == ";" { continue }
            var depth = 1
            var end = rest.startIndex
            while end < rest.endIndex {
                if rest[end] == "{" { depth += 1 }
                if rest[end] == "}" { depth -= 1 }
                if depth == 0 { break }
                end = rest.index(after: end)
            }
            guard end < rest.endIndex else { return }
            let body = rest[..<end]
            rest = rest[rest.index(after: end)...]
            guard !selectors.trimmingCharacters(in: .whitespacesAndNewlines).hasPrefix("@") else { continue }
            let declarations = SVGElementStyle.declarations(in: body)
            for selector in selectors.split(separator: ",") {
                let key = selector.trimmingCharacters(in: .whitespacesAndNewlines)
                guard let prefix = key.first, prefix == "." || prefix == "#",
                      key.count > 1, key.dropFirst().allSatisfy({ $0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" }) else {
                    continue
                }
                guard remainingSelectors > 0 else { throw SVGLoader.Failure.size }
                remainingSelectors -= 1
                rules[key, default: [:]].merge(declarations) { _, latest in latest }
            }
        }
    }

    private static func withoutComments(_ text: String) -> String {
        var remaining = text[...]
        var result = ""
        while let start = remaining.range(of: "/*") {
            result.append(contentsOf: remaining[..<start.lowerBound])
            remaining = remaining[start.upperBound...]
            guard let end = remaining.range(of: "*/") else { return result }
            remaining = remaining[end.upperBound...]
        }
        result.append(contentsOf: remaining)
        return result
    }

    func declarations(for attributes: [String: String]) -> [String: String] {
        var result: [String: String] = [:]
        for value in attributes["class"]?.split(whereSeparator: \.isWhitespace) ?? [] {
            if let rule = rules["." + value] { result.merge(rule) { _, latest in latest } }
        }
        if let id = attributes["id"], let rule = rules["#" + id] {
            result.merge(rule) { _, latest in latest }
        }
        return result
    }
}
