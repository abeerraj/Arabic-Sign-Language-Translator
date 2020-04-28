#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

// Protocol for callback action
@protocol CvCameraDelegate <NSObject>

- (void)matchedItem;
@end

// Public interface for camera. ViewController only needs to init, start and stop.
@interface OpenCVWrapper : NSObject

//functions prototypes
- (id)initWithController:(UIViewController<CvCameraDelegate>*)c andImageView:(UIImageView*)iv;
-(void)switchCamera;//Switch Camera
-(void)start;
-(void)stop;
-(bool)run;
-(int) trainSVM;

-(NSString*)returnstring;
@end
