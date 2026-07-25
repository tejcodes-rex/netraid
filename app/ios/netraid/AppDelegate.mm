#import "AppDelegate.h"

#import <React/RCTBundleURLProvider.h>

@implementation AppDelegate

- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions
{
  self.moduleName = @"netraid";
  // Custom initial props passed down to the React Native root component.
  // Launching with "--demo" (used by the iOS CI simulator run) opens the
  // camera-free Pipeline Demo directly, no deep-link confirmation dialog.
  BOOL demoMode = [[[NSProcessInfo processInfo] arguments] containsObject:@"--demo"];
  self.initialProps = demoMode ? @{@"demo": @YES} : @{};

  return [super application:application didFinishLaunchingWithOptions:launchOptions];
}

- (NSURL *)sourceURLForBridge:(RCTBridge *)bridge
{
  return [self bundleURL];
}

- (NSURL *)bundleURL
{
#if DEBUG
  return [[RCTBundleURLProvider sharedSettings] jsBundleURLForBundleRoot:@"index"];
#else
  return [[NSBundle mainBundle] URLForResource:@"main" withExtension:@"jsbundle"];
#endif
}

@end
