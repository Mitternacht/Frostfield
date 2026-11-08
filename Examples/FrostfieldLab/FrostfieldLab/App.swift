import UIKit

@main
final class AppDelegate: UIResponder, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        configurationForConnecting connectingSceneSession: UISceneSession,
        options: UIScene.ConnectionOptions
    ) -> UISceneConfiguration {
        UISceneConfiguration(name: "Default Configuration", sessionRole: connectingSceneSession.role)
    }
}

/// Each window scene owns its own render host. Nothing here is a process-wide
/// singleton, so two scenes can have different sizes and lifecycle states.
///
/// The delegate forwards its own scene's lifecycle transitions to that
/// scene's host, which pauses and resumes drawing per scene.
final class SceneDelegate: UIResponder, UIWindowSceneDelegate {
    var window: UIWindow?
    private var host: SceneViewController?

    func scene(
        _ scene: UIScene,
        willConnectTo session: UISceneSession,
        options connectionOptions: UIScene.ConnectionOptions
    ) {
        guard let windowScene = scene as? UIWindowScene else { return }
        let host = SceneViewController()
        let window = UIWindow(windowScene: windowScene)
        window.rootViewController = host
        window.makeKeyAndVisible()
        self.window = window
        self.host = host
    }

    func sceneWillEnterForeground(_ scene: UIScene) {
        host?.sceneActivityDidChange(.inactive)
    }

    func sceneDidBecomeActive(_ scene: UIScene) {
        host?.sceneActivityDidChange(.active)
    }

    func sceneWillResignActive(_ scene: UIScene) {
        host?.sceneActivityDidChange(.inactive)
    }

    func sceneDidEnterBackground(_ scene: UIScene) {
        host?.sceneActivityDidChange(.background)
    }
}
