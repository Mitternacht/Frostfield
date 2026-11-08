import MetalKit
import UIKit

/// An `MTKView` whose drawable size its owner sets from this view's own bounds
/// and the display scale of this view's trait collection, never from a
/// screen-wide value. Each window scene has its own instance, so two windows
/// on different displays or with different sizes size independently.
final class SceneRenderView: MTKView {
    /// Called at the end of every layout pass with the view's bounds size and
    /// the display scale of its current trait collection.
    var layoutDidChange: ((_ boundsSize: CGSize, _ displayScale: CGFloat) -> Void)?

    override init(frame: CGRect, device: (any MTLDevice)?) {
        super.init(frame: frame, device: device)
        // The owner sets `drawableSize`; MTKView must not derive it from
        // `contentScaleFactor` on its own.
        autoResizeDrawable = false
    }

    @available(*, unavailable)
    required init(coder: NSCoder) {
        fatalError("SceneRenderView is created in code")
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        // Reading `displayScale` inside `layoutSubviews` lets UIKit's trait
        // tracking lay this view out again when the scale changes, for
        // example when its window moves to another display.
        layoutDidChange?(bounds.size, traitCollection.displayScale)
    }
}
